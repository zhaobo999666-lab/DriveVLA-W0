#!/usr/bin/env python3
"""Qwen3-VL-8B 投影 MFU/HFU（对应第二份文档）。

把本文件与 calc_qwen3vl_mfu.py 放在同一目录。HFU 只增加实际启用
checkpoint 的文本/ViT block 投影估算；不重算 LM head 或视觉 merger。
"""

import argparse
import math

if __package__:
    from .calc_qwen3vl_mfu import add_inputs, check_peak, estimate, print_breakdown
else:
    from calc_qwen3vl_mfu import add_inputs, check_peak, estimate, print_breakdown


def replay_flops(result, text_skip, vit_skip, gc_enabled, ross_replay=0.0):
    if not 0 <= text_skip <= result.text_layers or not 0 <= vit_skip <= result.vit_layers:
        raise ValueError("跳过重算的层数不能小于 0 或超过对应模块层数")
    if not math.isfinite(ross_replay) or ross_replay < 0:
        raise ValueError("ROSS 重算 FLOPs 必须是有限的非负数")
    if ross_replay and (not gc_enabled or not any(p.forward for p in result.parts[4:])):
        raise ValueError("补计 ROSS 重算时须开启 GC 并同时提供 ROSS 正常前向投影 FLOPs")
    text, vit = result.parts[:2]
    return (
        text.forward * (result.text_layers - text_skip) / result.text_layers
        if gc_enabled and not text.frozen else 0,
        vit.forward * (result.vit_layers - vit_skip) / result.vit_layers
        if gc_enabled and not vit.frozen else 0,
        ross_replay,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_inputs(parser)
    parser.add_argument("--peak-tflops", type=float, default=432, help="单卡峰值假设（第二份文档默认 432）")
    parser.add_argument("--text-gc-skip", type=int, default=0, help="文本不重算层数；例：最后两层不重算填 2")
    parser.add_argument("--vit-gc-skip", type=int, default=0, help="视觉不重算层数；视觉完全不重算填 27")
    parser.add_argument("--no-gc", action="store_true", help="全部关闭重算")
    parser.add_argument("--ross-replay-flops", type=float, default=0.0, help="可选：已单独计数的 ROSS Linear 重算")
    a = parser.parse_args()
    try:
        result = estimate(a)
        check_peak(a.peak_tflops)
        text, vit, ross = replay_flops(result, a.text_gc_skip, a.vit_gc_skip, not a.no_gc, a.ross_replay_flops)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print_breakdown(result)
    if result.demo and (a.no_gc or a.text_gc_skip or a.vit_gc_skip or a.freeze_text or a.freeze_vit):
        print("提示：改变了示例的重算或冻结设置，步时仍为原采集值；仅作公式对照。")
    print(f"重算投影 FLOPs：文本 {text:.6e}；视觉 {vit:.6e}；ROSS {ross:.6e}")
    print("LM head 和视觉 merger/DeepStack 不参与重算；冻结 blocks 默认不重算。")
    print("单卡峰值假设 (TFLOP/s)     投影 MFU    投影 HFU（估算）")
    for peak in dict.fromkeys((a.peak_tflops, 486.354)):
        denom = peak * 1e12 * result.step_time
        print(f"{peak:>21g}  {result.useful/denom:>10.2%}  "
              f"{(result.useful+text+vit+ross)/denom:>13.2%}")


if __name__ == "__main__":
    main()
