"""Compare per-rank Insight step timing without mixing different workloads."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def collect(root, expected_ranks):
    meta = dict(l.split("=", 1) for l in (root / "run-info.txt").read_text().splitlines())
    if meta.get("job", "profile") != "profile":
        raise ValueError(f"{root}: expected an Insight profile run")
    by_rank = {}
    for f in root.rglob("step_trace_time.csv"):
        dbs = list(f.parent.glob("ascend_pytorch_profiler_*.db"))
        if len(dbs) != 1:
            raise ValueError(f"{f}: need the per-rank profiler DB")
        rank = int(dbs[0].stem.rsplit("_", 1)[-1])
        if rank in by_rank:
            raise ValueError(f"{root}: duplicate profiler data for rank {rank}")
        with f.open(newline="", encoding="utf-8-sig") as s:
            rows = list(csv.DictReader(s))
        if len(rows) != 1:
            raise ValueError(f"{f}: expected the single captured step")
        row = rows[0]
        step = int(row["Step"])
        comm = json.loads((f.parent / "communication.json").read_text(encoding="utf-8"))
        if f"step{step}" not in comm:
            raise ValueError(f"{f}: no communication data for step {step}")
        group = comm[f"step{step}"]
        hccl_ms = sum(
            float(op["Communication Time Info"]["Elapse Time(ms)"])
            for ops in group.values()
            for key, op in ops.items()
            if key != "Total Op Info" and isinstance(op, dict) and key.startswith("hcom_")
        )
        if float(row["Communication"]) > 0 and hccl_ms <= 0:
            raise ValueError(f"{f}: communication is nonzero but no HCCL operation durations were found")
        by_rank[rank] = (step, float(row["Bubble"])/1000,
                         float(row["Communication(Not Overlapped)"])/1000,
                         float(row["Communication"])/1000, hccl_ms, float(row["Stage"])/1000)
    if sorted(by_rank) != list(range(expected_ranks)):
        raise ValueError(f"{root}: expected {expected_ranks} ranks, got {sorted(by_rank)}")
    steps = {v[0] for v in by_rank.values()}
    if len(steps) != 1:
        raise ValueError(f"{root}: ranks captured different steps: {steps}")
    return meta, steps.pop(), by_rank


def describe(single, dual=None):
    sm, step, s = collect(single, 8)
    if dual is not None:
        dm, dstep, d = collect(dual, 16)
        if step != dstep or not sm.get("scaling_workload") or sm["scaling_workload"] != dm.get("scaling_workload"):
            raise ValueError("The 8/16-rank traces must use the same code/settings and capture step")
    else:
        d = None
    print(f"Insight step {step} (ms/step per rank; median [min, max])\n")
    print("| Metric | 8 NPUs | 16 NPUs |")
    print("|---|---:|---:|")

    def summary(data, index):
        if data is None:
            return "N/A"
        values = [v[index] for v in data.values()]
        return f"{median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"

    for i, name in enumerate(("Bubble", "CommNotOvl", "Communication total (timeline)",
                              "HCCL total (sum of collective/p2p op elapsed times)", "Stage"), 1):
        print(f"| {name} | {summary(s, i)} | {summary(d, i)} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("single", type=Path)
    p.add_argument("dual", type=Path, nargs="?")
    a = p.parse_args()
    describe(a.single, a.dual)
