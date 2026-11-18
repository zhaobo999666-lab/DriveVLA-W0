#!/usr/bin/env python3
"""Qwen3-VL-8B 投影 MFU（对应第一份文档，不含重算）。

不传参数时复算 A53+A54 Insight 的 rank 0、第 28 步。实际任务请传入
单卡形状和实测 --step-time；统计范围不含注意力 QKᵀ/AV、卷积或通信。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


TEXT_8B = dict(num_hidden_layers=36, hidden_size=4096, intermediate_size=12288,
               num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=151936)
VISION_8B = dict(depth=27, hidden_size=1152, intermediate_size=4304, spatial_merge_size=2,
                 out_hidden_size=4096, deepstack_visual_indexes=[8, 16, 24])


@dataclass
class Projection:
    name: str
    forward: int | float
    frozen: bool = False

    @property
    def useful(self):
        return self.forward if self.frozen else 3 * self.forward


@dataclass
class Result:
    parts: list[Projection]
    step_time: float
    demo: bool
    rows: int
    head_rows: int
    patches: int
    vocab: int
    text_layers: int
    vit_layers: int
    fused: int
    text_from_hook: bool
    text_frozen: bool
    vit_frozen: bool

    @property
    def useful(self):
        return sum(p.useful for p in self.parts)


def add_inputs(parser):
    parser.add_argument("--config", type=Path, help="Qwen3-VL config.json 或模型目录；默认 8B Instruct")
    parser.add_argument("--vocab-size", type=int, help="LM head 实际输出宽度；权重 resize 后可覆盖配置")
    parser.add_argument("--step-time", type=float, help="实测稳态秒/步")
    parser.add_argument("--text-rows", type=int, help="单卡文本投影总行数，含实际参与投影的 padding")
    parser.add_argument("--mbs", type=int, help="每卡 batch，和 --seq-len 一起用，替代 --text-rows")
    parser.add_argument("--seq-len", type=int, help="每卡 batch padding 后的长度")
    parser.add_argument("--vit-patches", type=int, help="单卡所有图像在 merger 前的 patch 数；纯文本传 0")
    head = parser.add_mutually_exclusive_group()
    head.add_argument("--lm-head-rows", type=int, help="实际送入 LM head 的行数（valid-token CE 用此参数）")
    head.add_argument("--full-lm-head", action="store_true", help="LM head 确实投影所有文本行时才使用")
    parser.add_argument("--text-hook-forward", type=float, help="仅文本 block 的 Linear hook 前向 FLOPs，不含 LM head")
    parser.add_argument("--fused-gateup", action="store_true", help="标记 gate/up 融合；用 hook 时补回该项")
    parser.add_argument("--fused-qkv", action="store_true", help="标记 QKV 融合；用 hook 时补回该项")
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--freeze-vit", action="store_true")
    parser.add_argument("--freeze-lm-head", action="store_true")
    parser.add_argument("--ross-linear-trainable-forward", type=float, default=0.0,
                        help="可选：单卡 ROSS 可训练 Linear 的前向 FLOPs；不填则不计 ROSS")
    parser.add_argument("--ross-linear-frozen-forward", type=float, default=0.0,
                        help="可选：单卡 ROSS 冻结 Linear 的前向 FLOPs；不含卷积和注意力 score")


def load_config(path):
    if path is None:
        return TEXT_8B, VISION_8B
    if path.is_dir():
        path /= "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("配置必须是 Qwen3-VL（model_type=qwen3_vl）")
    return config["text_config"], config["vision_config"]


def estimate(a):
    demo = all(v is None for v in (a.config, a.vocab_size, a.step_time, a.text_rows, a.mbs, a.seq_len,
                                   a.vit_patches, a.lm_head_rows, a.text_hook_forward)) and not (
        a.full_lm_head or a.ross_linear_trainable_forward or a.ross_linear_frozen_forward
    )
    if demo:
        rows, head_rows, patches, step_time = 42552, 318, 164736, 12.31222626
    else:
        if a.step_time is None or a.vit_patches is None:
            raise ValueError("自定义任务须提供 --step-time 和 --vit-patches（纯文本传 0）")
        if a.text_rows is None:
            if a.mbs is None or a.seq_len is None or min(a.mbs, a.seq_len) <= 0:
                raise ValueError("须提供 --text-rows，或同时提供正数 --mbs 和 --seq-len")
            rows = a.mbs * a.seq_len
        else:
            if a.mbs is not None or a.seq_len is not None:
                raise ValueError("--text-rows 与 --mbs/--seq-len 二选一")
            rows = a.text_rows
        head_rows = rows if a.full_lm_head else a.lm_head_rows
        if head_rows is None:
            raise ValueError("须提供 --lm-head-rows；确实投影全序列时用 --full-lm-head")
        patches, step_time = a.vit_patches, a.step_time
    if rows <= 0 or not 0 <= head_rows <= rows or patches < 0:
        raise ValueError("文本行数须为正，0 <= LM head 行数 <= 文本行数，patch 数须非负")
    if not math.isfinite(step_time) or step_time <= 0:
        raise ValueError("步时必须是有限的正数")
    costs = (a.text_hook_forward, a.ross_linear_trainable_forward, a.ross_linear_frozen_forward)
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in costs):
        raise ValueError("输入的前向 FLOPs 必须是有限的非负数")

    text, vision = load_config(a.config)
    h, inter, layers = (text[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers"))
    vocab = text["vocab_size"] if a.vocab_size is None else a.vocab_size
    heads, kv_heads = text["num_attention_heads"], text["num_key_value_heads"]
    if min(heads, kv_heads) <= 0:
        raise ValueError("注意力头数须为正")
    head_dim = text.get("head_dim") or h // heads
    q, kv = heads * head_dim, kv_heads * head_dim
    v, vi, vl, merge, out = (vision[k] for k in
                             ("hidden_size", "intermediate_size", "depth", "spatial_merge_size", "out_hidden_size"))
    if min(h, inter, layers, vocab, q, kv, v, vi, vl, merge, out) <= 0:
        raise ValueError("配置里的投影尺寸和层数须为正")
    unit = merge**2
    if patches % unit:
        raise ValueError(f"视觉 patch 数须能被 merger 的 {unit} 整除")

    qkv = 2 * rows * h * (q + 2 * kv) * layers
    o_proj = 2 * rows * q * h * layers
    gateup = 2 * rows * h * (2 * inter) * layers
    down = 2 * rows * inter * h * layers
    fused = (gateup if a.fused_gateup or demo else 0) + (qkv if a.fused_qkv else 0)
    text_from_hook = a.text_hook_forward is not None
    text_blocks = a.text_hook_forward + fused if text_from_hook else qkv + o_proj + gateup + down
    vit_blocks = 2 * patches * (4 * v * v + 2 * v * vi) * vl
    merger_width = v * unit
    merger_count = 1 + len(vision.get("deepstack_visual_indexes") or [])
    merger = 2 * (patches // unit) * merger_width * (merger_width + out) * merger_count
    parts = [
        Projection("文本 blocks", text_blocks, a.freeze_text),
        Projection("ViT blocks", vit_blocks, a.freeze_vit),
        Projection("ViT merger / DeepStack", merger, a.freeze_vit),
        Projection("LM head", 2 * head_rows * h * vocab, a.freeze_lm_head),
        Projection("ROSS Linear（可训练）", a.ross_linear_trainable_forward),
        Projection("ROSS Linear（冻结）", a.ross_linear_frozen_forward, True),
    ]
    return Result(parts, step_time, demo, rows, head_rows, patches, vocab, layers, vl, fused, text_from_hook,
                  a.freeze_text, a.freeze_vit)


def print_breakdown(result):
    if result.demo:
        print("示例：A53+A54 Insight 基线 rank 0、第 28 步；单卡单步，非 16 卡稳态平均。")
    print(f"文本：{result.text_layers} 层、{result.rows:,} 行；LM head：{result.head_rows:,} 行 × {result.vocab:,}")
    print(f"视觉：{result.vit_layers} 层、{result.patches:,} 个 merger 前 patch")
    mode = "已补到 hook" if result.text_from_hook else "已包含在形状公式中"
    print(f"文本融合投影前向：{result.fused:.6e} FLOPs（{mode}）")
    for p in result.parts:
        print(f"{p.name}: 前向 {p.forward:.6e} × {1 if p.frozen else 3} = {p.useful:.6e} FLOPs")
    print(f"正常训练投影 FLOPs：{result.useful:.6e}；步时：{result.step_time:.6f} s")
    print("范围：Linear/融合投影；不计 QKᵀ/AV、patch embedding 卷积或其他 ROSS 算子。")


def check_peak(peak):
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("单卡峰值假设必须是有限的正数")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_inputs(parser)
    parser.add_argument("--peak-tflops", type=float, default=486.354, help="单卡峰值假设（第一份文档默认 486.354）")
    a = parser.parse_args()
    try:
        result = estimate(a)
        check_peak(a.peak_tflops)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print_breakdown(result)
    print(f"单卡峰值假设 {a.peak_tflops:g} TFLOP/s；投影 MFU = "
          f"{result.useful / (result.step_time * a.peak_tflops * 1e12):.2%}")


if __name__ == "__main__":
    main()
