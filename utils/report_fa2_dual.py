"""Compare the clean 16-rank GC2 runs and their three MFU conventions."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median, quantiles

if __package__:
    from .report_projection_mfu import read_run
else:
    from report_projection_mfu import read_run


def read_metrics(path):
    info, _, wall, runtime, shapes = read_run(path, "measure")
    if info["world_size"] != "16":
        raise ValueError("Expected 16 ranks")
    nav = path / "train-fast/navsim"
    state = json.loads((nav / "trainer_state.json").read_text())
    losses = [h["loss"] for h in state["log_history"] if "loss" in h]
    if len(losses) != int(info["steps"]) or not all(math.isfinite(x) for x in losses):
        raise ValueError(f"{path}: invalid loss history")
    tokens = sum(math.prod(b["input_ids"]) for rank in shapes for b in rank)
    ranks = sorted((nav / "peak-memory").glob("rank-*.json"))
    if ranks and len(ranks) != 16:
        raise ValueError(f"{path}: incomplete peak memory records")
    peak = (max(json.loads(f.read_text())["max_allocated_bytes"] for f in ranks)/1e9 if ranks else None)
    rank0_log = (path / "train-a53.log").read_text(errors="replace")
    rank0 = re.findall(r"peak HBM allocated \(rank0\): ([\d.]+) GB", rank0_log)
    return info, (runtime, mean(wall), median(wall), quantiles(wall, n=20, method="inclusive")[18],
                  int(info["gbs"])/mean(wall), int(info["steps"])*int(info["gbs"])/runtime,
                  tokens/sum(wall), float(rank0[-1]) if rank0 else None, peak, losses[-1]), shapes


def methods(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = [x.strip() for x in line.strip("|").split("|")]
        if len(vals) == 5 and vals[0].startswith(("Document ", "MindSpeed ")):
            name, _, _, mfu, hfu = vals
            result[name] = (mfu, hfu)
    if len(result) != 5:
        raise ValueError(f"{path}: missing three-method table")
    return result


def compare(baseline, optimized, baseline_report, optimized_report=None):
    a, x, sa = read_metrics(baseline)
    b, y, sb = read_metrics(optimized)
    keys = ("revision", "mbs", "gbs", "steps", "warmup", "fast_gateup", "fast_zero", "gc_skip_text_last")
    if any(a[k] != b[k] for k in keys) or b.get("fa2_unpad_cache") != "1":
        raise ValueError("Runs do not match the intended settings")
    if sa != sb:
        raise ValueError("Steady rank/step batch shapes differ")
    print(f"# A53+A54 16 卡：GC2 基线与 FA2 缓存\n\n{baseline.name} → {optimized.name}\n")
    print("稳态：第 11–50 步，按每步最慢 rank 的含等待用时。TPS 含输入 padding。\n")
    print("| 指标 | GC2 基线 | FA2 缓存 |\n|---|---:|---:|")
    labels = ("50 步 Trainer runtime (s)", "平均步时 (s)", "P50 (s)", "P95 (s)", "稳态样本/s",
              "端到端样本/s", "稳态输入 token/s", "框架 HBM rank0 (GB)", "框架 HBM 全卡最高 (GB)", "末步 loss")
    for label, av, bv in zip(labels, x, y):
        show = lambda v: "未记录" if v is None else f"{v:,.4f}"
        print(f"| {label} | {show(av)} | {show(bv)} |")
    print(f"\n平均步时减少 {x[1]-y[1]:+.4f} s（{1-y[1]/x[1]:+.2%}）；单次对照不等于可重复收益。\n")
    if optimized_report is not None:
        a, b = methods(baseline_report), methods(optimized_report)
        if a.keys() != b.keys():
            raise ValueError("MFU conventions differ")
        print("| 口径 | 基线 MFU / HFU | 优化 MFU / HFU |\n|---|---:|---:|")
        for key in a:
            print(f"| {key} | {' / '.join(a[key])} | {' / '.join(b[key])} |")
        print("\nHFU 含重算，不能当作有效 MFU；分母沿用各口径假设值。")
    else:
        print("三口径 MFU/HFU 等待优化版计数轮；本表仅使用已完成的 50 步正式测速。\n")
    stats = sorted((optimized / "train-fast/navsim/fa2-unpad").glob("rank-*.json"))
    if len(stats) != 16:
        raise ValueError("Expected cache statistics on all 16 ranks")
    rows = [json.loads(f.read_text()) for f in stats]
    if not all(r["cache"] and r["hits"] > 0 and r["builds"] > 0 for r in rows):
        raise ValueError("Cache did not hit on every rank")
    print(f"缓存 50 步每 rank 中位数：准备 {median(r['builds'] for r in rows):g} 次，"
          f"命中 {median(r['hits'] for r in rows):g} 次。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "optimized", "baseline_report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("optimized_report", type=Path, nargs="?")
    compare(**vars(parser.parse_args()))
