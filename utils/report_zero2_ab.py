#!/usr/bin/env python3
"""Compare same-workload dual-node ZeRO-2 timing trials."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


REFERENCE_STEP_S = 12.2517
REFERENCE_EFFECTIVE_MFU = 0.4807
REFERENCE_HFU = 0.6123


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def read_info(run: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (run / "run-info.txt").read_text().splitlines())


def read_trial(run: Path) -> dict[str, object]:
    info = read_info(run)
    if info.get("job") != "measure" or info.get("world_size") != "16":
        raise ValueError(f"{run}: expected a completed 16-rank measure run")
    for node in ("a53", "a54"):
        if (run / f"train-{node}.exit").read_text().strip() != "0":
            raise ValueError(f"{run}: {node} did not exit successfully")

    config = json.loads((run / "deepspeed-zero2.json").read_text())["zero_optimization"]
    expected_bucket: str | int = info["zero2_bucket_size"]
    if expected_bucket != "auto":
        expected_bucket = int(expected_bucket)
    if config["stage"] != 2 or config["reduce_bucket_size"] != expected_bucket:
        raise ValueError(f"{run}: generated bucket does not match run-info")
    if config["overlap_comm"] != (info["zero2_overlap_comm"] == "1"):
        raise ValueError(f"{run}: generated overlap_comm does not match run-info")

    steps, warmup = int(info["steps"]), int(info["warmup"])
    world, mbs = int(info["world_size"]), int(info["mbs"])
    rank_walls: list[list[float]] = []
    peak_bytes: list[int] = []
    for rank in range(world):
        wall_path = run / "train-fast/navsim/step-wall" / f"rank-{rank:02}.json"
        rows = json.loads(wall_path.read_text())
        selected = [row["wall_s"] for row in rows if warmup < row["step"] <= steps]
        if len(selected) != steps - warmup or any(not math.isfinite(x) or x <= 0 for x in selected):
            raise ValueError(f"{wall_path}: incomplete or invalid steady window")
        rank_walls.append(selected)
        memory = json.loads(
            (run / "train-fast/navsim/peak-memory" / f"rank-{rank:02}.json").read_text()
        )["max_allocated_bytes"]
        peak_bytes.append(int(memory))

    critical = [max(values) for values in zip(*rank_walls)]
    mean_s = statistics.mean(critical)
    state = json.loads((run / "train-fast/navsim/trainer_state.json").read_text())
    losses = [row["loss"] for row in state["log_history"] if "loss" in row]
    if state["global_step"] != steps or not losses or not all(math.isfinite(float(x)) for x in losses):
        raise ValueError(f"{run}: incomplete run or non-finite loss")

    same_work = REFERENCE_STEP_S / mean_s
    return {
        "run": run.name,
        "bucket": info["zero2_bucket_size"],
        "overlap": info["zero2_overlap_comm"],
        "deterministic": info["hccl_deterministic"],
        "gc_skip": info["gc_skip_text_last"],
        "mean": mean_s,
        "p50": statistics.median(critical),
        "p95": percentile(critical, 0.95),
        "throughput": world * mbs / mean_s,
        "effective_mfu": REFERENCE_EFFECTIVE_MFU * same_work,
        "hfu": REFERENCE_HFU * same_work,
        "hbm": max(peak_bytes) / 1e9,
        "loss": float(losses[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [read_trial(path) for path in args.runs]
    baseline = rows[0]["mean"]

    lines = [
        "# ZeRO-2 双机短测对比",
        "",
        "同一模型、数据、MBS 24、GC2 和 16 卡工作量。步时取每一步 16 个 rank 中最慢者，再对稳态窗口统计。",
        "投影 MFU/HFU 以已审计基线 12.2517 s、48.07%/61.23% 按相同步工作量缩放；候选胜出后仍需独立 audit 确认，HFU 不能作为有效 MFU 验收值。",
        "",
        "| 试验 | bucket 元素 | overlap | deterministic | 步时均值 | P50 | P95 | 相对基线 | 吞吐 samples/s | 投影有效 MFU | 投影 HFU | 最大 HBM | 末次 loss |",
        "|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = (float(row["mean"]) / float(baseline) - 1) * 100
        lines.append(
            f"| {row['run']} | {row['bucket']} | {row['overlap']} | {row['deterministic']} | "
            f"{row['mean']:.4f}s | {row['p50']:.4f}s | {row['p95']:.4f}s | {delta:+.2f}% | "
            f"{row['throughput']:.2f} | {row['effective_mfu']:.2%} | {row['hfu']:.2%} | "
            f"{row['hbm']:.2f}GB | {row['loss']:.4f} |"
        )
    best = min(rows, key=lambda row: float(row["mean"]))
    lines.extend([
        "",
        f"墙钟最优候选：`{best['run']}`，稳态关键路径均值 {best['mean']:.4f} s。",
        "仅当 loss 有限、全 rank HBM 安全，并在重复 50 步与独立 audit 中复现后，才升级为正式配置。",
        "",
    ])
    report = "\n".join(lines)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
