"""Validate and summarize a same-node FA2 unpadding A/B."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

if __package__:
    from .report_projection_mfu import read_run
else:
    from report_projection_mfu import read_run


def get_records(root, info, enabled):
    nav = root / "train-fast/navsim"
    memory, builds, hits = [], [], []
    for rank in range(int(info["world_size"])):
        fname = f"rank-{rank:02}.json"
        hbm = json.loads((nav / "peak-memory" / fname).read_text())["max_allocated_bytes"]
        stats = json.loads((nav / "fa2-unpad" / fname).read_text())
        if hbm <= 0 or stats["cache"] != enabled or stats["builds"] <= 0:
            raise ValueError(f"{root}: invalid FA2 or HBM stats on rank {rank}")
        if (enabled and stats["hits"] <= 0) or (not enabled and stats["hits"] != 0):
            raise ValueError(f"{root}: unexpected FA2 cache hits on rank {rank}")
        memory.append(hbm / 1e9)
        builds.append(stats["builds"])
        hits.append(stats["hits"])
    history = json.loads((nav / "trainer_state.json").read_text())["log_history"]
    losses = {h["step"]: (h["loss"], h["grad_norm"]) for h in history if "loss" in h}
    if set(losses) != set(range(1, int(info["steps"])+1)) or any(
        not math.isfinite(v) for pair in losses.values() for v in pair
    ):
        raise ValueError(f"{root}: incomplete or non-finite loss/gradient history")
    return memory, builds, hits, losses


def compare(base, cached):
    a, _, aw, ar, ash = read_run(base, "measure")
    b, _, bw, br, bsh = read_run(cached, "measure")
    suffix = " fa2_cache="
    if (a["node"], b["node"], a["world_size"], b["world_size"]) != ("a53", "a53", "8", "8"):
        raise ValueError("Expected two A53 8-rank trials")
    keys = ("revision", "mbs", "gbs", "steps", "warmup", "fast_gateup", "fast_zero", "gc_skip_text_last")
    if any(a[k] != b[k] for k in keys) or (a["fa2_unpad_cache"], b["fa2_unpad_cache"]) != ("0", "1"):
        raise ValueError("Training settings differ beyond the cache switch")
    if not (a["scaling_workload"].endswith(suffix+"0") and b["scaling_workload"].endswith(suffix+"1")
            and a["scaling_workload"][:-1] == b["scaling_workload"][:-1]):
        raise ValueError("Training snapshot, data or switches do not match")
    if any(i["fa2_unpad_stats"] != "1" or i["fa2_unpad_markers"] != "0" for i in (a,b)):
        raise ValueError("Require counters on and profiler markers off in both runs")
    if ash != bsh:
        raise ValueError("Per-rank batch shapes differ in the steady window")
    ah, ab, ac, al = get_records(base, a, False)
    bh, bb, bc, bl = get_records(cached, b, True)
    print(f"# A53 FA2 A/B: {base.name} vs {cached.name}")
    print(f"Same snapshot/inputs shapes; steady steps {int(a['warmup'])+1}–{a['steps']}.\n")
    print("| Metric | Cache off | Cache on |")
    print("|---|---:|---:|")
    for label, av, bv in (
        ("Mean steady step, s", mean(aw), mean(bw)),
        ("P50 steady step, s", median(aw), median(bw)),
        ("Steady samples/s", int(a["gbs"])/mean(aw), int(b["gbs"])/mean(bw)),
        ("Trainer runtime, s", ar, br),
        ("Peak allocated HBM, max across 8 ranks, GB", max(ah), max(bh)),
        ("Metadata builds/rank (median)", median(ab), median(bb)),
        ("Cache hits/rank (median)", median(ac), median(bc)),
        ("Final loss", al[max(al)][0], bl[max(bl)][0]),
    ):
        print(f"| {label} | {av:.4f} | {bv:.4f} |")
    print(f"\nWall step reduction: {mean(aw)-mean(bw):+.4f} s ({1-mean(bw)/mean(aw):+.2%}).")
    print(f"Max absolute paired loss difference: {max(abs(al[k][0]-bl[k][0]) for k in al):.4f}.")
    print("This is one short A/B; confirm a repeatable gain with a 50-step test before promoting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("cached", type=Path)
    args = parser.parse_args()
    compare(args.baseline, args.cached)
