#!/usr/bin/env python3
"""复算《DriveVLA-W0_MFU计算公式与现状分析.md》。

所有 FLOPs 都是单卡、单步；步时单位为秒，峰值单位为 TFLOP/s。
仅按文档统计权重投影，不计注意力 score、重算、通信或优化器。
"""

import argparse
import math


def projection_flops(rows, hidden, intermediate, vocab, q_dim, kv_dim, head_rows):
    """返回每个投影的前向 FLOPs。层内项是单层，LM head 是整个模型一次。"""
    if not 0 <= head_rows <= rows:
        raise ValueError("LM head 行数必须在 0 与总 token 行数之间")
    return {
        "Q proj": 2 * rows * hidden * q_dim,
        "K proj": 2 * rows * hidden * kv_dim,
        "V proj": 2 * rows * hidden * kv_dim,
        "O proj": 2 * rows * q_dim * hidden,
        "gate/up": 2 * rows * hidden * (2 * intermediate),
        "down": 2 * rows * intermediate * hidden,
        "LM head": 2 * head_rows * hidden * vocab,
    }


def calculate_mfu(trainable_forward, frozen_forward, extra, step_time, peak_tflops):
    """返回训练 FLOPs、实际 TFLOP/s 和 MFU 比值。"""
    if any(not math.isfinite(v) or v < 0 for v in (trainable_forward, frozen_forward, extra)):
        raise ValueError("前向 FLOPs 必须是有限的非负数")
    if any(not math.isfinite(v) or v <= 0 for v in (step_time, peak_tflops)):
        raise ValueError("步时和峰值算力必须是有限的正数")
    flops = 3 * (trainable_forward + extra) + frozen_forward
    achieved = flops / step_time / 1e12
    return flops, achieved, achieved / peak_tflops


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-time", type=float, help="实测稳态秒/步；文档示例为 6.85")
    parser.add_argument("--peak-tflops", type=float, default=486.354, help="单卡峰值假设（默认 486.354）")
    shape = parser.add_argument_group("按 Emu3 模型形状计算（默认）")
    shape.add_argument("--mbs", type=int, default=32, help="每卡 micro batch（默认 32）")
    shape.add_argument("--seq-len", type=int, default=1280, help="每样本序列长度（默认 1280）")
    shape.add_argument("--layers", type=int, default=32)
    shape.add_argument("--hidden-size", type=int, default=4096)
    shape.add_argument("--intermediate-size", type=int, default=14336)
    shape.add_argument("--vocab-size", type=int, default=184622)
    shape.add_argument("--q-dim", type=int, default=4096)
    shape.add_argument("--kv-dim", type=int, default=1024)
    shape.add_argument("--lm-head-rows", type=int, help="单卡实际投影的行数（默认 MBS × 序列长度）")
    logs = parser.add_argument_group("直接输入日志的单卡前向 FLOPs")
    logs.add_argument("--trainable-forward-flops", type=float, help="hook 的可训练投影前向 FLOPs，不含 extra")
    logs.add_argument("--frozen-forward-flops", type=float, default=0.0, help="冻结投影前向 FLOPs")
    logs.add_argument("--extra-flops", type=float, default=0.0, help="hook 漏掉的可训练投影前向 FLOPs")
    a = parser.parse_args()
    try:
        parts = None
        trainable = a.trainable_forward_flops
        if trainable is None:
            if a.extra_flops != 0 or a.frozen_forward_flops != 0:
                raise ValueError("形状模式已覆盖投影；输入日志补偿时须同时提供 --trainable-forward-flops")
            dims = (a.mbs, a.seq_len, a.layers, a.hidden_size, a.intermediate_size, a.vocab_size, a.q_dim, a.kv_dim)
            if any(v <= 0 for v in dims):
                raise ValueError("batch、序列长度和模型尺寸必须为正整数")
            rows = a.mbs * a.seq_len
            head_rows = rows if a.lm_head_rows is None else a.lm_head_rows
            parts = projection_flops(rows, a.hidden_size, a.intermediate_size, a.vocab_size,
                                     a.q_dim, a.kv_dim, head_rows)
            layer_flops = sum(v for k, v in parts.items() if k != "LM head")
            trainable = layer_flops * a.layers + parts["LM head"]
        elif a.step_time is None:
            raise ValueError("使用自己的 FLOPs 日志时必须传入 --step-time")
        step_time = 6.85 if a.step_time is None else a.step_time
        flops, achieved, mfu = calculate_mfu(trainable, a.frozen_forward_flops, a.extra_flops,
                                            step_time, a.peak_tflops)
    except ValueError as exc:
        parser.error(str(exc))

    if parts is not None:
        print(f"模型形状：层数 {a.layers}，文本行数 {rows}，LM head 行数 {head_rows}")
        for name, value in parts.items():
            print(f"  {name:>8}: {value:.6e} FLOPs")
        print(f"层内投影小计（单层前向）：{layer_flops:.6e} FLOPs")
    print(f"可训练前向：{trainable:.6e} FLOPs")
    print(f"冻结前向：  {a.frozen_forward_flops:.6e} FLOPs")
    print(f"可训练补算：{a.extra_flops:.6e} FLOPs")
    print(f"训练 FLOPs = 3 × (可训练前向 + 补算) + 冻结前向 = {flops:.6e}")
    time_source = "（沿用文档示例）" if a.step_time is None else ""
    print(f"步时：{step_time:.6f} s{time_source}；单卡峰值假设：{a.peak_tflops:g} TFLOP/s")
    print(f"实际投影吞吐：{achieved:.3f} TFLOP/s")
    print(f"MFU = {mfu:.4%}（一位小数：{mfu:.1%}）")


if __name__ == "__main__":
    main()
