"""Report matching projection MFU/HFU and optionally 8-to-16 NPU scaling."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


def read_run(root, job):
    info = dict(line.split("=", 1) for line in (root / "run-info.txt").read_text().splitlines())
    if info["job"] != job:
        raise ValueError(f"{root}: expected job={job}")
    world = int(info["world_size"])
    if world not in (8, 16, 32):
        raise ValueError(f"{root}: expected 8, 16 or 32 ranks")
    node_text = info.get("node_list", "")
    if not node_text and world == 8:
        node_text = info["node"]
    elif not node_text and world == 16:
        node_text = "a53,a54"
    elif not node_text and world == 32:
        node_text = "a51,a52,a53,a54"
    nodes = tuple(node_text.split(","))
    if len(nodes) != world // 8 or any(not node for node in nodes):
        raise ValueError(f"{root}: node_list does not match world_size={world}")
    for node in nodes:
        if (root / f"train-{node}.exit").read_text().strip() != "0":
            raise ValueError(f"{root}: training failed on {node}")
    state = json.loads((root / "train-fast/navsim/trainer_state.json").read_text())
    timings = {h["step"]: h["step_time_full_s"] for h in state["log_history"] if "step_time_full_s" in h}
    steps, warmup = int(info["steps"]), int(info["warmup"])
    if state["global_step"] != steps or set(timings) != set(range(steps)):
        raise ValueError(f"{root}: incomplete training or step times")
    steady = [timings[i] for i in range(warmup, steps)]
    runtime = next(h["train_runtime"] for h in reversed(state["log_history"]) if "train_runtime" in h)
    if not steady or any(not math.isfinite(t) or t <= 0 for t in [*steady, runtime]):
        raise ValueError(f"{root}: invalid timing")
    if warmup < 1:
        raise ValueError("At least one warmup step is required for step-end wall time")
    wall_by_rank, shapes_by_rank = [], []
    for rank in range(world):
        path = root / "train-fast/navsim/step-wall" / f"rank-{rank:02}.json"
        rows = json.loads(path.read_text())
        if [r["step"] for r in rows] != list(range(2, steps + 1)):
            raise ValueError(f"{path}: incomplete wall-clock time")
        window = [r for r in rows if r["step"] > warmup]
        walls = [r["wall_s"] for r in window]
        if any(not math.isfinite(t) or t <= 0 for t in walls):
            raise ValueError(f"{path}: invalid wall-clock time")
        shapes = [r["shapes"] for r in window]
        if any(s.get("input_ids", [None])[0] != int(info["mbs"]) or not s.get("pixel_values") for s in shapes):
            raise ValueError(f"{path}: missing or unexpected batch shapes")
        wall_by_rank.append(walls)
        shapes_by_rank.append(shapes)
    return info, steady, [max(step) for step in zip(*wall_by_rank)], runtime, shapes_by_rank


def read_projections(root, info):
    steps, warmup, world, batch = (int(info[k]) for k in ("steps", "warmup", "world_size", "mbs"))
    if world * batch != int(info["gbs"]):
        raise ValueError("Expected gradient accumulation 1")
    totals = {}
    for rank in range(world):
        path = root / "train-fast/navsim/projection-audit" / f"rank-{rank:02}.json"
        rows = json.loads(path.read_text())
        if [r["step"] for r in rows] != list(range(steps)) or any(r["samples"] != batch for r in rows):
            raise ValueError(f"{path}: incomplete or unexpected batches")
        for row in rows[warmup:]:
            groups = row["groups"]
            if any(not groups.get(g, {}).get("trainable") for g in ("text", "lm_head")):
                raise ValueError(f"{path}: missing text or LM head projection")
            if info["fast_gateup"] == "1" and not groups["text"].get("fused_forward"):
                raise ValueError(f"{path}: fused gate/up requested but not observed")
            if groups["lm_head"].get("replay"):
                raise ValueError(f"{path}: unexpected LM head replay")
            for group, counts in groups.items():
                total = totals.setdefault(group, {})
                for name, value in counts.items():
                    if not isinstance(value, int) or value < 0:
                        raise ValueError(f"{path}: invalid projection count")
                    total[name] = total.get(name, 0) + value
    return totals


def print_scaling(single, dual, dual_run=None):
    s, _, sw, sr, _ = read_run(single, "measure")
    d, _, dw, dr, _ = read_run(dual, "measure") if dual_run is None else dual_run
    if (s["world_size"], d["world_size"]) != ("8", "16"):
        raise ValueError("Scaling requires an 8-rank baseline and a 16-rank measurement")
    keys = ("revision", "mbs", "steps", "warmup", "fast_gateup", "fast_zero", "gc_skip_text_last",
            "scaling_workload")
    if any(k not in s or k not in d or s[k] != d[k] for k in keys):
        raise ValueError("Scaling requires the same code, MBS, model/data, training settings and timing window")
    sg, dg = int(s["gbs"]), int(d["gbs"])
    if sg != 8 * int(s["mbs"]) or dg != 16 * int(d["mbs"]):
        raise ValueError("Scaling expects GBS = world size × MBS (gradient accumulation 1)")
    total_steps = int(s["steps"])
    print(f"\n8→16 NPU scaling: {single.name} → {dual.name}; MBS {s['mbs']}, GBS {sg} → {dg}")
    print(f"Steady optimizer steps: {int(s['warmup']) + 1}–{total_steps}; "
          f"mean wall step 8 / 16 NPUs: {mean(sw):.3f} / {mean(dw):.3f} s")
    print("| Window | 8 NPUs (samples/s) | 16 NPUs (samples/s) | Speedup | Scaling efficiency |")
    print("|---|---:|---:|---:|---:|")
    for name, small, large in (("Steady wall", sg / mean(sw), dg / mean(dw)),
                               ("End-to-end", sg * total_steps / sr, dg * total_steps / dr)):
        speedup = large / small
        print(f"| {name} | {small:.2f} | {large:.2f} | {speedup:.3f}× | {speedup/2:.2%} |")


def mindspeed_costs(measure, info, ross_file):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    if __package__:
        from .flops_mindspeed import Qwen3VLFlopsCounter
    else:
        from flops_mindspeed import Qwen3VLFlopsCounter

    shapes = json.loads((measure / "shapes.json").read_text())
    config = Qwen3VLConfig.from_pretrained(shapes["model"], local_files_only=True)
    counter = Qwen3VLFlopsCounter(config)
    samples = shapes["samples"]
    if not samples or any(not 0 <= s["labelled"] <= s["llm"] for s in samples):
        raise ValueError("MindSpeed comparison requires real shapes and valid label counts")
    skip = int(info["gc_skip_text_last"])
    useful, executed = [
        mean(counter.estimate_flops([s["llm"]], step_time=1, images_seqlens=s["vit"],
                                    include_recompute=replay, gc_skip_text_last=skip)
             for s in samples)
        for replay in (False, True)
    ]
    ross = float(json.loads(ross_file.read_text())["total_ross_useful"])
    if not math.isfinite(ross) or ross <= 0:
        raise ValueError("Invalid ROSS FLOPs")
    ce = (6 * config.text_config.hidden_size * config.text_config.vocab_size
          * mean(s["llm"] - (s["labelled"] or s["llm"]) for s in samples))
    return {
        "MindSpeed backbone": (useful, executed),
        "MindSpeed + ROSS": (useful + ross, executed + ross),
        "MindSpeed + ROSS, valid-token CE": (useful + ross - ce, executed + ross - ce),
    }


def report(measure, audit, single=None, ross_file=None):
    dual_run = read_run(measure, "measure")
    clean, timing, wall, runtime, shapes = dual_run
    counted, _, audit_wall, _, audit_shapes = read_run(audit, "audit")
    keys = ("workload", "world_size", "mbs", "gbs", "steps", "warmup", "fast_gateup", "fast_zero",
            "gc_skip_text_last")
    if any(clean[k] != counted[k] for k in keys):
        raise ValueError("Measurement and audit workloads do not match")
    if shapes != audit_shapes:
        raise ValueError("Measurement and audit rank/step batch shapes do not match")
    totals = read_projections(audit, counted)
    model = sum(3 * g.get("trainable", 0) + g.get("frozen", 0) for g in totals.values())
    replay = sum(g.get("replay", 0) for g in totals.values())
    world = int(clean["world_size"])
    print(f"Runs: {measure.name} (clean), {audit.name} (projection audit)")
    print(f"Steady optimizer steps: {int(clean['warmup']) + 1}–{clean['steps']}; {world} ranks")
    print(f"Clean wall mean/P50: {mean(wall):.3f}/{median(wall):.3f} s; "
          f"end-to-end: {int(clean['gbs']) * int(clean['steps']) / runtime:.2f} samples/s")
    print(f"Legacy compute/optimizer mean (excludes data wait): {mean(timing):.3f} s")
    print(f"Audit wall mean: {mean(audit_wall):.3f} s; audit vs clean: {mean(audit_wall)/mean(wall)-1:+.2%}")
    print("\n| Group | Useful FLOPs/card/step | Replay FLOPs/card/step | Fused forward FLOPs/card/step |")
    print("|---|---:|---:|---:|")
    scale = world * len(timing)
    for name, g in sorted(totals.items()):
        print(f"| {name} | {(3*g.get('trainable', 0)+g.get('frozen', 0))/scale:.6e} | "
              f"{g.get('replay', 0)/scale:.6e} | {g.get('fused_forward', 0)/scale:.6e} |")
    print(f"\nMean LM-head rows/card/step: {totals['lm_head']['rows']/scale:.2f}")
    print("\n| Assumed peak per NPU | Projection MFU | Projection HFU estimate | Audit-run MFU |")
    print("|---|---:|---:|---:|")
    for peak in (432, 486, 486.354):
        denom = peak * 1e12 * world * sum(wall)
        audit_denom = peak * 1e12 * world * sum(audit_wall)
        print(f"| {peak} TFLOP/s | {model/denom:.2%} | {(model+replay)/denom:.2%} | "
              f"{model/audit_denom:.2%} |")
    print("\nProjection-only; excludes attention-score matmuls and convolutions. "
          "HFU adds observed replayed projections; backward FLOPs are estimated. "
          "The peak values are assumptions, not hardware counter measurements.")
    if ross_file is not None:
        print("\nThree calculation methods, same clean steady wall window:")
        print("| Method | Peak TFLOP/s | Useful FLOPs/card/step | Useful MFU | HFU estimate |")
        print("|---|---:|---:|---:|---:|")
        for label, peak, include_hfu in (("Document 1: projections", 486.354, False),
                                         ("Document 2: projections", 432, True)):
            denom = peak * 1e12 * mean(wall)
            hfu = f"{(model+replay)/scale/denom:.2%}" if include_hfu else "—"
            print(f"| {label} | {peak:g} | {model/scale:.6e} | {model/scale/denom:.2%} | {hfu} |")
        denom = 486e12 * mean(wall)
        for label, (useful, executed) in mindspeed_costs(measure, clean, ross_file).items():
            batch = int(clean["mbs"])
            print(f"| {label} | 486 | {useful*batch:.6e} | {useful*batch/denom:.2%} | {executed*batch/denom:.2%} |")
        print("MindSpeed HFU adds backbone replay only; ROSS cost excludes replay. "
              "The valid-token CE row changes only the LM-head estimate.")
    if single is not None:
        print_scaling(single, measure, dual_run)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("measure", type=Path, help="8- or 16-rank measure run")
    p.add_argument("audit", type=Path, nargs="?", help="matching projection audit run")
    p.add_argument("--single", type=Path, help="matching 8-rank measure run; omit audit for scaling only")
    p.add_argument("--ross-costs", type=Path, help="ROSS FLOPs JSON; add the three-method comparison")
    args = p.parse_args()
    if args.audit is not None:
        report(args.measure, args.audit, args.single, args.ross_costs)
    elif args.single is not None:
        print_scaling(args.single, args.measure)
    else:
        p.error("provide an audit run, --single, or both")


if __name__ == "__main__":
    main()
