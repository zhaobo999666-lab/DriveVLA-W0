#!/usr/bin/env python3
"""Save auditable FLOPs estimates and compare completed benchmark runs."""

import argparse
import json
import re
import statistics
from pathlib import Path


def count_ross(model, sd):
    """Count the isolated ROSS forward on CPU with its training tensor shapes."""
    import torch
    from diffusers import AutoencoderKL
    from diffusers.models import activations
    from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross import (
        unet_2d_condition as unet_module,
    )
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from flops_mindspeed import _measure_forward_flops

    torch.set_num_threads(4)
    # The Ascend Diffusers build selects GEGLU by package presence, even on CPU.
    # Use its native CPU branch for this counting process only.
    if hasattr(activations, "is_torch_npu_available"):
        activations.is_torch_npu_available = lambda: False
    cfg = Qwen3VLConfig.from_pretrained(model, local_files_only=True)
    unet = unet_module.UNet2DConditionModel.from_pretrained(
        str(Path(sd) / "unet"), local_files_only=True,
    ).float().eval()
    vae = AutoencoderKL.from_pretrained(
        str(Path(sd) / "vae"), local_files_only=True,
    ).float().eval()
    enc = torch.load(Path(unet_module.__file__).with_name("negative_prompt_sd15.pt"),
                     map_location="cpu", weights_only=True).float()
    h, w = 35, 63
    width = int(unet.config.block_out_channels[0])
    sample = torch.zeros(1, 4, h, w)
    cond = torch.zeros(1, width, h, w)
    time = torch.tensor([500])
    unet_flops = _measure_forward_flops(
        unet, lambda: unet(sample, time, enc, return_dict=False, z=cond),
        dense_coeff=3.0, attn_coeff=3.5,
    )
    vae_flops = _measure_forward_flops(
        vae, lambda: vae.encode(torch.zeros(1, 3, 280, 504)),
        dense_coeff=1.0, attn_coeff=1.0,
    )
    # Two trainable two-layer projectors, one spatial and one action token.
    projection = 6 * (h * w + 1) * (cfg.text_config.hidden_size * width + width**2)
    return {
        "unet_useful": unet_flops,
        "vae_forward": vae_flops,
        "condition_projectors": projection,
        "total_ross_useful": unet_flops + vae_flops + projection,
        "latent_hw": [h, w],
        "condition_shape": list(enc.shape),
        "method": "CPU forward hooks, analytical training multipliers; no ROSS replay",
    }


def summarize(runs, ross, peak, ce_shapes=None):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from flops_mindspeed import Qwen3VLFlopsCounter

    print("| Run | ZeRO | MBS | Steady steps | Mean s/step | Backbone useful | Useful + ROSS | "
          "Backbone + replay | Useful + ROSS + backbone replay | End-to-end samples/s |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    scores = []
    adjusted = []
    for run in runs:
        info = dict(line.split("=", 1) for line in (run / "run-info.txt").read_text().splitlines())
        step, warmup, batch, gbs = (int(info[k]) for k in ("steps", "warmup", "mbs", "gbs"))
        skip = int(info.get("gc_skip_text_last", "0"))
        if skip:
            log = (run / "train-fast.log").read_text(errors="replace")
            if not re.search(rf"\[gc\] skipping checkpoint on last {skip}/\d+ decoder layers", log):
                raise ValueError(f"{run}: did not verify text replay setting in the training log")
        shapes = json.loads((run / "shapes.json").read_text())
        config = Qwen3VLConfig.from_pretrained(shapes["model"], local_files_only=True)
        counter = Qwen3VLFlopsCounter(config)
        backbone, replay = (statistics.mean(counter.estimate_flops(
            [row["llm"]], step_time=1.0, images_seqlens=row["vit"], include_recompute=recompute,
            gc_skip_text_last=skip,
        ) for row in shapes["samples"]) for recompute in (False, True))
        state = json.loads((run / "train-fast/navsim/trainer_state.json").read_text())
        hist = state["log_history"]
        timing = {h["step"]: h["step_time_full_s"] for h in hist if "step_time_full_s" in h}
        if state["global_step"] != step or set(timing) != set(range(step)):
            raise ValueError(f"{run}: training or timing data incomplete")
        mean = statistics.mean(timing[i] for i in range(warmup, step))
        runtime = next(h["train_runtime"] for h in reversed(hist) if "train_runtime" in h)
        scale = batch / mean / peak
        vals = [cost * scale for cost in (backbone, backbone + ross, replay, replay + ross)]
        print(f"| {run.name} | {info['fast_zero']} | {batch} | {warmup+1}–{step} | {mean:.3f} | "
              + " | ".join(f"{v:.2%}" for v in vals) + f" | {gbs * step / runtime:.2f} |")
        scores.append((vals[1], run.name))
        if ce_shapes is not None:
            labelled = ce_shapes["samples"]
            keys = ("model", "data", "data_root")
            if (any(shapes[k] != ce_shapes[k] for k in keys)
                    or [{k: r[k] for k in ("index", "llm", "vit")} for r in labelled]
                    != [{k: r[k] for k in ("index", "llm", "vit")} for r in shapes["samples"]]):
                raise ValueError(f"{run}: valid-token CE audit samples do not match")
            skipped = statistics.mean(r["llm"] - (r["labelled"] or r["llm"]) for r in labelled)
            delta = 6 * config.text_config.hidden_size * config.text_config.vocab_size * skipped
            adjusted.append((run.name, (backbone + ross - delta) * scale, delta * scale))
    best = max(scores)
    print(f"\nBest tested useful MFU including ROSS: **{best[0]:.2%}** ({best[1]}).")
    if adjusted:
        print("\nValid-token CE sensitivity (only the LM head changes; assumes this path is active):")
        for name, val, delta in adjusted:
            print(f"- {name}: useful + ROSS {val:.2%} ({-100 * delta:+.2f} pp)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count-ross")
    count.add_argument("model")
    count.add_argument("sd")
    count.add_argument("output", type=Path)
    table = sub.add_parser("table")
    table.add_argument("costs", type=Path)
    table.add_argument("runs", nargs="+", type=Path)
    table.add_argument("--peak", type=float, default=486e12)
    table.add_argument("--ce-shapes", type=Path, help="same shapes with shifted label counts for the CE audit")
    args = p.parse_args()
    if args.command == "count-ross":
        result = json.dumps(count_ross(args.model, args.sd), indent=2) + "\n"
        args.output.write_text(result)
        print(result, end="")
    else:
        if args.peak <= 0:
            table.error("--peak must be positive")
        ce_shapes = json.loads(args.ce_shapes.read_text()) if args.ce_shapes else None
        summarize(args.runs, json.loads(args.costs.read_text())["total_ross_useful"], args.peak, ce_shapes)


if __name__ == "__main__":
    main()
