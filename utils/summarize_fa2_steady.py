"""Summarize per-rank and per-node steady-state Insight CSV data."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles


METRICS = (
    "stage_s",
    "compute_s",
    "comm_total_s",
    "comm_unoverlapped_s",
    "free_s",
    "bubble_s",
    "hccl_elapsed_s",
    "largest_free_s",
)


def p95(values):
    return quantiles(values, n=20, method="inclusive")[18]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    by_rank = defaultdict(list)
    with args.csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rank = int(row["rank"])
            by_rank[rank].append(
                {
                    "optimizer_step": int(row["optimizer_step"]),
                    **{name: float(row[name]) for name in METRICS},
                }
            )

    ranks = sorted(by_rank)
    if ranks != list(range(16)):
        raise ValueError(f"Expected ranks 0-15, found {ranks}")
    if any(len(by_rank[rank]) != 40 for rank in ranks):
        raise ValueError("Expected 40 steady steps for every rank")

    print("| Rank | Node | Stage median | Stage P95 | CommNotOvl median | "
          "Free median | HCCL median | Slowest step |")
    print("|---:|---|---:|---:|---:|---:|---:|---|")
    for rank in ranks:
        rows = by_rank[rank]
        slowest = max(rows, key=lambda row: row["stage_s"])
        print(
            f"| {rank} | {'A53' if rank < 8 else 'A54'} "
            f"| {median(r['stage_s'] for r in rows):.6f} "
            f"| {p95([r['stage_s'] for r in rows]):.6f} "
            f"| {median(r['comm_unoverlapped_s'] for r in rows):.6f} "
            f"| {median(r['free_s'] for r in rows):.6f} "
            f"| {median(r['hccl_elapsed_s'] for r in rows):.6f} "
            f"| {slowest['optimizer_step']} ({slowest['stage_s']:.6f} s) |"
        )

    print("\n| Scope | Stage median | Stage P95 | CommNotOvl median | "
          "Free median | HCCL median |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, selected in (
        ("A53 ranks 0-7", ranks[:8]),
        ("A54 ranks 8-15", ranks[8:]),
        ("All ranks", ranks),
    ):
        rows = [row for rank in selected for row in by_rank[rank]]
        print(
            f"| {label} "
            f"| {median(r['stage_s'] for r in rows):.6f} "
            f"| {p95([r['stage_s'] for r in rows]):.6f} "
            f"| {median(r['comm_unoverlapped_s'] for r in rows):.6f} "
            f"| {median(r['free_s'] for r in rows):.6f} "
            f"| {median(r['hccl_elapsed_s'] for r in rows):.6f} |"
        )


if __name__ == "__main__":
    main()
