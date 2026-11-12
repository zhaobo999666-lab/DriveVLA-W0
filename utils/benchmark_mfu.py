#!/usr/bin/env python3
"""Sample NAVSIM training shapes on CPU and summarize the single-node A5 benchmark."""

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

PEAK = 486e12  # Assumed BF16 FLOP/s per NPU; keep fixed across comparisons.
DEVICES = 8
WARMUP = 5
STEPS = 50


def sample_shapes(model, data, rgb, output, count):
    import torch
    from qwenvl.dataset.data_qwen_vla import make_supervised_data_module_navsim2_vla_ross
    from qwenvl.train.argument import DataArguments
    from qwenvl.utils.token_utils import VLA_SPECIAL_TOKENS
    from reference.transformers.src.transformers.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLProcessor,
    )
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": sorted(
        set(tokenizer.additional_special_tokens) | set(VLA_SPECIAL_TOKENS.values())
    )})
    for name, token in VLA_SPECIAL_TOKENS.items():
        setattr(tokenizer, name, token)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 4096
    args = DataArguments(
        dataset_type="navsim2va_ross", data_path=data, data_root=rgb,
        max_pixels=903168, min_pixels=12544, cur_frame_idx=3,
        action_tokenizer_path=str(Path.cwd() / "configs" / "fast"),
    )
    args.model_type = "qwen3vl_ross"
    args.image_processor = Qwen3VLProcessor.from_pretrained(model, local_files_only=True).image_processor
    args.image_processor.size = {"longest_edge": args.max_pixels, "shortest_edge": args.min_pixels}
    module = make_supervised_data_module_navsim2_vla_ross(tokenizer=tokenizer, data_args=args)
    dataset, collate = module["train_dataset"], module["data_collator"]
    indices = random.Random(42).sample(range(len(dataset)), min(count, len(dataset)))
    if not indices:
        raise ValueError("Training data is empty")
    rows = []
    for index in indices:
        batch = collate([dataset[index]])
        llm = int(batch["attention_mask"].sum().item())
        vit = batch["image_grid_thw"].prod(dim=-1).tolist()
        if not llm or len(vit) != 2 or min(vit) <= 0:
            raise ValueError(f"Unexpected training shape for sample {index}: {llm=}, {vit=}")
        labelled = int(batch["labels"][..., 1:].ne(-100).sum().item())
        rows.append({"index": index, "llm": llm, "vit": vit, "labelled": labelled})

    payload = {"model": model, "data": data, "data_root": rgb, "samples": rows}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Sampled {len(rows)} training items with real dataset/collator: {output}")
    print(f"Median valid LLM tokens: {statistics.median(r['llm'] for r in rows)}")
    print(f"Median supervised positions: {statistics.median(r['labelled'] for r in rows)}")
    print(f"Median ViT patches (previous/current): "
          f"{[statistics.median(r['vit'][i] for r in rows) for i in (0, 1)]}")


def report(run, batch, profiles, steps, warmup, devices):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from utils.flops_mindspeed import Qwen3VLFlopsCounter

    shape = json.loads((run / "shapes.json").read_text(encoding="utf-8"))
    counter = Qwen3VLFlopsCounter(Qwen3VLConfig.from_pretrained(shape["model"], local_files_only=True))
    # Average per-sample FLOPs, preserving the squared-length attention terms.
    flop = statistics.mean(counter.estimate_flops(
        batch_seqlens=[r["llm"]], images_seqlens=r["vit"], step_time=1.0,
    ) for r in shape["samples"])
    gbs = batch * devices
    print(f"Qwen3-VL backbone useful FLOPs/sample (mean of {len(shape['samples'])} items): {flop:.6e}")
    print(f"Trainval: {shape['data']} | {devices} NPUs | MBS={batch} GBS={gbs}")
    print("Peak assumption: 486 TFLOP/s per NPU. MFU excludes ROSS and recompute in the numerator.")
    print(f"Steady window: optimizer steps {warmup + 1}-{steps}. Full step time excludes waiting for data.")
    print(f"End-to-end uses all {steps} steps from Trainer's train_runtime.")

    rates = {}
    for profile in profiles:
        state = json.loads((run / f"train-{profile}/navsim/trainer_state.json").read_text(encoding="utf-8"))
        hist = state["log_history"]
        # RossTrainer logs before the optimizer step, so its step field is zero-based.
        timing = {h["step"]: h["step_time_full_s"] for h in hist if "step_time_full_s" in h}
        if state["global_step"] != steps or set(timing) != set(range(steps)):
            raise ValueError(f"{profile}: expected a completed {steps}-step run and {steps} timing records")
        steady = [timing[i] for i in range(warmup, steps)]
        runtime = next(h["train_runtime"] for h in reversed(hist) if "train_runtime" in h)
        if not all(math.isfinite(t) and t > 0 for t in (*steady, runtime)):
            raise ValueError(f"{profile}: invalid timing record")
        median, mean = statistics.median(steady), statistics.mean(steady)
        throughput = gbs * steps / runtime
        rates[profile] = throughput
        peak = re.findall(r"peak HBM allocated \(rank0\): ([\d.]+) GB",
                          (run / f"train-{profile}.log").read_text(encoding="utf-8", errors="replace"))
        print(f"{profile}: step P50={median:.3f}s mean={mean:.3f}s, "
              f"useful MFU={flop * batch / (mean * PEAK):.2%}, "
              f"end-to-end MFU={flop * throughput / (devices * PEAK):.2%}, "
              f"end-to-end={throughput:.2f} samples/s, "
              f"rank0 peak HBM={peak[-1] + ' GB' if peak else 'unavailable'}")
    if {"safe", "fast"} <= rates.keys():
        print(f"Fast / safe end-to-end throughput: {rates['fast'] / rates['safe']:.3f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shapes = sub.add_parser("shapes")
    shapes.add_argument("model")
    shapes.add_argument("data")
    shapes.add_argument("rgb")
    shapes.add_argument("output", type=Path)
    shapes.add_argument("--count", type=int, default=32)
    summary = sub.add_parser("report")
    summary.add_argument("run", type=Path)
    summary.add_argument("batch", type=int)
    summary.add_argument("profiles", nargs="*", metavar="safe|fast")
    summary.add_argument("--steps", type=int, default=STEPS)
    summary.add_argument("--warmup", type=int, default=WARMUP)
    summary.add_argument("--devices", type=int, choices=(8, 16), default=DEVICES)
    args = parser.parse_args()
    if args.command == "shapes":
        sample_shapes(args.model, args.data, args.rgb, args.output, args.count)
    else:
        if any(p not in ("safe", "fast") for p in args.profiles):
            summary.error("profiles must be safe or fast")
        if not 0 <= args.warmup < args.steps:
            summary.error("expected 0 <= warmup < steps")
        report(args.run, args.batch, args.profiles or ("safe", "fast"), args.steps, args.warmup, args.devices)


if __name__ == "__main__":
    main()
