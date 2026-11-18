"""Summarize steady-state HCCL operations from MindStudio analysis databases."""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles


def step_number(value):
    text = str(value).removeprefix("step")
    return int(text) if text.isdigit() else None


def op_family(name):
    """Collapse profiler kernel suffixes into an HCCL operation family."""
    match = re.match(r"(hcom_[A-Za-z0-9]+)", name or "")
    return match.group(1) if match else (name or "-")


def profile_rank(profiler: Path) -> int:
    profile_dir = profiler.parent.parent
    info_files = sorted(profile_dir.glob("profiler_info*.json"))
    if not info_files:
        raise ValueError(f"No profiler_info JSON under {profile_dir}")
    with info_files[0].open(encoding="utf-8") as handle:
        return int(json.load(handle)["rank_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", default="10-49")
    args = parser.parse_args()
    start, end = (int(value) for value in args.steps.split("-", 1))

    rank_totals = {}
    call_samples = defaultdict(list)
    rank_step_totals = defaultdict(float)
    for profiler in sorted(args.root.rglob("analysis.db")):
        rank = profile_rank(profiler)
        with sqlite3.connect(
            (profiler.parent / "analysis.db").resolve().as_uri() + "?mode=ro",
            uri=True,
        ) as conn:
            rows = conn.execute(
                """
                SELECT hccl_op_name, group_name, type, step, elapse_time,
                       wait_time, transit_time, synchronization_time, idle_time
                FROM CommAnalyzerTime
                WHERE lower(hccl_op_name) LIKE 'hcom_%'
                """
            )
            totals = defaultdict(lambda: [0.0, 0, 0.0, 0.0, 0.0, 0.0])
            selected_steps = set()
            for name, group, op_type, step, elapsed, wait, transit, sync, idle in rows:
                ordinal = step_number(step)
                if ordinal is None or not start <= ordinal <= end:
                    continue
                selected_steps.add(ordinal)
                key = (op_family(name), group or "", op_type or "")
                values = totals[key]
                values[0] += elapsed or 0
                values[1] += 1
                values[2] += wait or 0
                values[3] += transit or 0
                values[4] += sync or 0
                values[5] += idle or 0
                call_samples[key].append(elapsed or 0)
                rank_step_totals[(rank, ordinal)] += elapsed or 0
            expected = set(range(start, end + 1))
            if selected_steps != expected:
                raise ValueError(f"Rank {rank}: incomplete steps {sorted(selected_steps)}")
            rank_totals[rank] = {
                key: tuple(value / len(expected) for value in values)
                for key, values in totals.items()
            }

    if not rank_totals:
        raise ValueError("No profiler databases found")
    keys = set().union(*(set(values) for values in rank_totals.values()))
    summaries = {}
    for key in keys:
        per_rank = [values[key] for values in rank_totals.values() if key in values]
        summaries[key] = tuple(
            median(values[index] for values in per_rank)
            for index in range(6)
        )

    print(f"{len(rank_totals)} ranks; steps {start}-{end} "
          "(raw STEP_TIME ids, excluding 10 warm-up steps)\n")
    print("| HCCL op | Group | Type | Elapsed ms/step | Calls/step | "
          "Call median ms | Call P95 ms | Call max ms | Wait ms | "
          "Transit ms | Sync ms | Idle ms |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (name, group, op_type), values in sorted(
        summaries.items(), key=lambda item: item[1][0], reverse=True
    ):
        samples = call_samples[(name, group, op_type)]
        print(
            f"| {name} | {group or '-'} | {op_type or '-'} "
            f"| {values[0]:.3f} | {values[1]:.2f} "
            f"| {median(samples):.3f} "
            f"| {quantiles(samples, n=20, method='inclusive')[18]:.3f} "
            f"| {max(samples):.3f} "
            f"| {values[2]:.3f} | {values[3]:.3f} "
            f"| {values[4]:.3f} | {values[5]:.3f} |"
        )

    print("\n| Slowest rank/step | HCCL elapsed ms |")
    print("|---|---:|")
    for (rank, step), elapsed in sorted(
        rank_step_totals.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        print(f"| rank {rank}, optimizer step {step + 1} | {elapsed:.3f} |")


if __name__ == "__main__":
    main()
