#!/usr/bin/env python3
"""复算《DriveVLA-W0 MFU-HFU 计算口径文档.md》的投影 MFU / HFU。

所有 FLOPs 都是单卡、单步。HFU 只给 checkpoint 层内的投影加重算，
LM head 只计一次正常训练。此处 HFU 是公式估算，不是硬件计数器读数。
"""

import argparse
import math


EXAMPLE_HOOK = 2.639e14
EXAMPLE_LAYER_EXTRA = 3.079e14
EXAMPLE_LM_HEAD = 6.190e13
REFERENCE_PEAK = 486.354
REFERENCE_MFU = 0.5615


def calculate_flops(hook, layer_extra, lm_head, frozen, layers, gc_skip, gc_enabled=True):
    """返回正常训练 FLOPs、重算 FLOPs、含重算总 FLOPs 和重算比例。"""
    if any(not math.isfinite(v) or v < 0 for v in (hook, layer_extra, lm_head, frozen)):
        raise ValueError("前向 FLOPs 必须是有限的非负数")
    if layers <= 0 or not 0 <= gc_skip <= layers:
        raise ValueError("要求层数 > 0 且 0 <= gc-skip <= layers")
    layer_forward = hook + layer_extra
    model = 3 * (layer_forward + lm_head) + frozen
    fraction = (layers - gc_skip) / layers if gc_enabled else 0.0
    replay = fraction * layer_forward
    return model, replay, model + replay, fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-time", type=float, help="实测稳态秒/步；默认仅示例按旧 MFU 56.15%% 反推")
    parser.add_argument("--peak-tflops", type=float, default=432, help="单卡峰值假设（默认 432）")
    parser.add_argument("--layers", type=int, default=32, help="总层数（默认 32）")
    parser.add_argument("--gc-skip", type=int, help="末尾不重算的层数（示例为 4，自定义日志默认 0）")
    parser.add_argument("--no-gc", action="store_true", help="关闭重算，此时 HFU 等于 MFU")
    parser.add_argument("--hook-flops", type=float, help="仅层内、可训练投影的原始前向 hook FLOPs")
    parser.add_argument("--frozen-flops", type=float, default=0.0, help="冻结投影前向 FLOPs")
    parser.add_argument("--layer-extra", type=float, help="层内融合漏计的前向 FLOPs，不含 LM head")
    parser.add_argument("--lmhead-extra", type=float, help="LM head 前向 FLOPs，只填一次")
    shape = parser.add_argument_group("从形状补算额外投影（与直接输入补算值二选一）")
    shape.add_argument("--tokens", type=int, help="单卡单步文本投影行数，用于融合 gate/up、QKV")
    shape.add_argument("--fused-gateup", action="store_true")
    shape.add_argument("--fused-qkv", action="store_true")
    shape.add_argument("--lm-head-rows", type=int, help="LM head 实际投影行数，仅用于计算该项")
    shape.add_argument("--hidden-size", type=int, default=4096)
    shape.add_argument("--intermediate-size", type=int, default=14336)
    shape.add_argument("--vocab-size", type=int, default=184622)
    shape.add_argument("--q-dim", type=int, default=4096)
    shape.add_argument("--kv-dim", type=int, default=1024)
    a = parser.parse_args()
    try:
        fused = a.fused_gateup or a.fused_qkv
        demo = a.hook_flops is None
        if demo and (fused or a.layer_extra is not None or a.lmhead_extra is not None
                     or a.lm_head_rows is not None or a.frozen_flops != 0):
            raise ValueError("计算自己的投影补偿时须提供 --hook-flops 和 --step-time")
        if not demo and a.step_time is None:
            raise ValueError("使用自己的 FLOPs 日志时必须传入 --step-time")
        if fused and (a.layer_extra is not None or a.tokens is None or a.tokens <= 0):
            raise ValueError("按形状补算融合项须传入正数 --tokens，并省略 --layer-extra")
        if a.lm_head_rows is not None and (a.lmhead_extra is not None or a.lm_head_rows < 0):
            raise ValueError("--lm-head-rows 须非负，并且不能和 --lmhead-extra 同用")
        if any(v <= 0 for v in (a.hidden_size, a.intermediate_size, a.vocab_size, a.q_dim, a.kv_dim)):
            raise ValueError("模型尺寸必须是正整数")

        hook = EXAMPLE_HOOK if demo else a.hook_flops
        layer_extra = EXAMPLE_LAYER_EXTRA if demo else (a.layer_extra or 0)
        lm_head = EXAMPLE_LM_HEAD if demo else (a.lmhead_extra or 0)
        if fused:
            width = (2 * a.intermediate_size if a.fused_gateup else 0) + (a.q_dim + 2 * a.kv_dim if a.fused_qkv else 0)
            layer_extra = 2 * a.tokens * a.hidden_size * width * a.layers
        if a.lm_head_rows is not None:
            lm_head = 2 * a.lm_head_rows * a.hidden_size * a.vocab_size
        gc_skip = (4 if demo else 0) if a.gc_skip is None else a.gc_skip
        model, replay, total, fraction = calculate_flops(hook, layer_extra, lm_head, a.frozen_flops,
                                                         a.layers, gc_skip, not a.no_gc)
        step_time = a.step_time
        if step_time is None:
            example_model = 3 * (EXAMPLE_HOOK + EXAMPLE_LAYER_EXTRA + EXAMPLE_LM_HEAD)
            step_time = example_model / (REFERENCE_MFU * REFERENCE_PEAK * 1e12)
        if any(not math.isfinite(v) or v <= 0 for v in (step_time, a.peak_tflops)):
            raise ValueError("步时和峰值算力必须是有限的正数")
    except ValueError as exc:
        parser.error(str(exc))

    print(f"层内 hook 前向：{hook:.6e} FLOPs")
    print(f"层内融合补算：  {layer_extra:.6e} FLOPs")
    print(f"LM head 前向：  {lm_head:.6e} FLOPs（不参与重算）")
    print(f"冻结前向：      {a.frozen_flops:.6e} FLOPs")
    print(f"正常训练 FLOPs：{model:.6e}")
    print(f"重算 FLOPs：    {replay:.6e}（层内前向 × {fraction:.3%}）")
    print(f"含重算 FLOPs：  {total:.6e}")
    time_source = "（由文档的旧 MFU 56.15% 反推）" if a.step_time is None else ""
    print(f"步时：{step_time:.6f} s{time_source}")
    print("单卡峰值假设 (TFLOP/s)     MFU       HFU")
    for peak in dict.fromkeys((a.peak_tflops, REFERENCE_PEAK)):
        denom = peak * 1e12 * step_time
        print(f"{peak:>21g}  {model/denom:>8.2%}  {total/denom:>8.2%}")


if __name__ == "__main__":
    main()
