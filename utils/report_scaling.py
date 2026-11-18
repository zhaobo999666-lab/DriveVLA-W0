#!/usr/bin/env python3
"""Build one report for matched 8-, 16-, and 32-NPU performance runs.

Measure/audit runs provide wall time, throughput, HBM, MFU, and HFU.
Profile runs provide Insight Step Trace, communication, free time, and HCCL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKOUT = ROOT / "checkout"
for path in (
    ROOT,
    CHECKOUT,
    CHECKOUT / "utils",
    CHECKOUT / "reference" / "transformers" / "src",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from report_projection_mfu import (  # noqa: E402
    mindspeed_costs,
    read_projections,
    read_run,
)


PROJECTION_PEAKS = {"document1": 486.354, "document2": 432.0}
MINDSPEED_PEAK = 486.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower)


def parse_scale(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def parse_run_info(root: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (root / "run-info.txt").read_text().splitlines())


def profile_metrics(root: Path, world_size: int) -> dict[str, float]:
    """Read one captured Insight step from every rank in a profile run."""
    info = parse_run_info(root)
    if info.get("job") != "profile":
        raise ValueError(f"{root}: expected job=profile")
    if int(info["world_size"]) != world_size:
        raise ValueError(f"{root}: profile world_size does not match {world_size}")

    rows: dict[int, dict[str, float]] = {}
    for csv_path in root.rglob("step_trace_time.csv"):
        dbs = list(csv_path.parent.glob("ascend_pytorch_profiler_*.db"))
        if len(dbs) != 1:
            raise ValueError(f"{csv_path}: expected one profiler database beside the CSV")
        rank = int(dbs[0].stem.rsplit("_", 1)[-1])
        if rank in rows:
            raise ValueError(f"{root}: duplicate profile data for rank {rank}")
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            trace_rows = list(csv.DictReader(stream))
        if len(trace_rows) != 1:
            raise ValueError(f"{csv_path}: expected one captured Step Trace row, got {len(trace_rows)}")
        trace = trace_rows[0]
        step = int(trace["Step"])
        communication = json.loads((csv_path.parent / "communication.json").read_text(encoding="utf-8"))
        group = communication.get(f"step{step}")
        if group is None:
            raise ValueError(f"{csv_path}: communication.json has no step {step}")
        hccl_ms = sum(
            float(operation["Communication Time Info"]["Elapse Time(ms)"])
            for operations in group.values()
            for key, operation in operations.items()
            if key != "Total Op Info" and key.startswith("hcom_") and isinstance(operation, dict)
        )
        rows[rank] = {
            "step": float(step),
            "computing_s": float(trace["Computing"]) / 1e6,
            "communication_not_overlapped_s": float(trace["Communication(Not Overlapped)"]) / 1e6,
            "communication_total_s": float(trace["Communication"]) / 1e6,
            "overlapped_communication_s": float(trace["Overlapped"]) / 1e6,
            "free_s": float(trace["Free"]) / 1e6,
            "preparing_s": float(trace["Preparing"]) / 1e6,
            "stage_s": float(trace["Stage"]) / 1e6,
            "bubble_s": float(trace["Bubble"]) / 1e6,
            "hccl_total_s": hccl_ms / 1e3,
        }
    if sorted(rows) != list(range(world_size)):
        raise ValueError(f"{root}: expected ranks 0..{world_size - 1}, got {sorted(rows)}")
    steps = {int(row["step"]) for row in rows.values()}
    if len(steps) != 1:
        raise ValueError(f"{root}: ranks captured different steps: {sorted(steps)}")

    result: dict[str, float] = {"profile_step": float(steps.pop())}
    for key in next(iter(rows.values())):
        if key == "step":
            continue
        values = [row[key] for row in rows.values()]
        result[f"{key}_median"] = statistics.median(values)
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
    return result


def measure_metrics(root: Path, ross_file: Path) -> dict[str, object]:
    clean, timing, wall, runtime, rank_shapes = read_run(root, "measure")
    world, batch = int(clean["world_size"]), int(clean["mbs"])
    gbs, steps = int(clean["gbs"]), int(clean["steps"])
    mean_wall = statistics.mean(wall)
    p50_wall = statistics.median(wall)
    p95_wall = percentile(wall, 0.95)
    steady_samples_s = gbs / mean_wall
    e2e_samples_s = gbs * steps / runtime
    shape_payload = json.loads((root / "shapes.json").read_text())
    mean_llm_tokens = statistics.mean(row["llm"] for row in shape_payload["samples"])

    peaks = []
    for rank in range(world):
        path = root / "train-fast/navsim/peak-memory" / f"rank-{rank:02}.json"
        if path.is_file():
            peaks.append(int(json.loads(path.read_text())["max_allocated_bytes"]))
    state = json.loads((root / "train-fast/navsim/trainer_state.json").read_text())
    losses = [row["loss"] for row in state["log_history"] if "loss" in row]
    if any(not math.isfinite(float(loss)) for loss in losses):
        raise ValueError(f"{root}: non-finite loss")

    return {
        "run": root.name,
        "nodes": len((clean.get("node_list") or clean["node"]).split(",")),
        "node_list": clean.get("node_list") or clean["node"],
        "world_size": world,
        "mbs": batch,
        "gbs": gbs,
        "steps": steps,
        "warmup": int(clean["warmup"]),
        "runtime_s": runtime,
        "trainer_step_mean_s": statistics.mean(timing),
        "step_wall_mean_s": mean_wall,
        "step_wall_p50_s": p50_wall,
        "step_wall_p95_s": p95_wall,
        "steady_samples_s": steady_samples_s,
        "e2e_samples_s": e2e_samples_s,
        "steady_tps": steady_samples_s * mean_llm_tokens,
        "e2e_tps": e2e_samples_s * mean_llm_tokens,
        "mean_llm_tokens_per_sample": mean_llm_tokens,
        "hbm_max_gb": max(peaks) / 1e9 if len(peaks) == world else None,
        "last_loss": float(losses[-1]),
        "_info": clean,
        "_timing": timing,
        "_wall": wall,
        "_shapes": rank_shapes,
    }


def calculation_metrics(measure: Path, audit: Path, ross_file: Path) -> dict[str, float]:
    clean, timing, wall, _, _ = read_run(measure, "measure")
    counted, _, audit_wall, _, _ = read_run(audit, "audit")
    matching = ("workload", "world_size", "mbs", "gbs", "steps", "warmup",
                "fast_gateup", "fast_zero", "gc_skip_text_last")
    if any(clean[key] != counted[key] for key in matching):
        raise ValueError(f"{measure} and {audit}: workload mismatch")
    world, batch = int(clean["world_size"]), int(clean["mbs"])
    mean_wall = statistics.mean(wall)
    mean_audit_wall = statistics.mean(audit_wall)
    totals = read_projections(audit, counted)
    scale = world * len(wall)
    useful = sum(3 * group.get("trainable", 0) + group.get("frozen", 0) for group in totals.values()) / scale
    replay = sum(group.get("replay", 0) for group in totals.values()) / scale

    result = {
        "projection_useful_per_card_step": useful,
        "projection_replay_per_card_step": replay,
        "audit_vs_measure_step_ratio": mean_audit_wall / mean_wall,
    }
    for label, peak in PROJECTION_PEAKS.items():
        denominator = peak * 1e12 * mean_wall
        result[f"{label}_mfu"] = useful / denominator
        result[f"{label}_hfu"] = (useful + replay) / denominator

    mindspeed = mindspeed_costs(measure, clean, ross_file)
    for label, (useful_flops, executed_flops) in mindspeed.items():
        key = label.lower().replace(" ", "_").replace("+", "plus").replace(",", "")
        denominator = MINDSPEED_PEAK * 1e12 * mean_wall
        result[f"{key}_mfu"] = useful_flops * batch / denominator
        result[f"{key}_hfu"] = executed_flops * batch / denominator
    return result


def build_scale(
    name: str,
    measure: Path,
    audit: Path,
    profile: Path,
    ross_file: Path,
) -> dict[str, object]:
    row = measure_metrics(measure, ross_file)
    row.update(calculation_metrics(measure, audit, ross_file))
    row.update(profile_metrics(profile, int(row["world_size"])))
    row = {key: value for key, value in row.items() if not key.startswith("_")}
    row["scale"] = name
    return row


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_ms(seconds: float) -> str:
    """Format a duration stored in seconds as milliseconds."""
    return f"{seconds * 1000:.1f}"


def render(rows: list[dict[str, object]], reference: dict[str, float]) -> str:
    by_scale = {str(row["scale"]): row for row in rows}
    single = rows[0]
    scale_names = {"single": "单机", "dual": "双机", "quad": "四机"}
    machine_labels = [
        f"{scale_names.get(str(row['scale']), str(row['scale']))} {row['world_size']} 卡"
        for row in rows
    ]

    def table_header(first_column: str) -> list[str]:
        return [
            "| " + " | ".join([first_column, *machine_labels]) + " |",
            "|---" + "|---:" * len(rows) + "|",
        ]

    def metric_row(label: str, values: list[str]) -> str:
        return "| " + " | ".join([label, *values]) + " |"

    lines = [
        "# DriveVLA-W0：单机、双机与四机性能对比",
        "",
        "本报告使用相同模型、数据、MBS、BF16、FA2、ZeRO-2、GC2 和热身窗口，"
        "仅改变节点数。正式步时和 TPS 来自关闭 Profiler 的 measure 轮；"
        "Step Trace、通信、Free 和 HCCL 来自独立 profile 轮，不与正式 MFU 混用。",
        "",
        "## 1. 端到端吞吐与步时",
        "",
    ]
    lines.extend(table_header("指标"))
    lines.extend([
        metric_row("MBS / GBS", [f"{row['mbs']} / {row['gbs']}" for row in rows]),
        metric_row("time_elapsed (ms)", [fmt_ms(float(row["runtime_s"])) for row in rows]),
        metric_row("Step wall 均值 (ms)", [fmt_ms(float(row["step_wall_mean_s"])) for row in rows]),
        metric_row("P50 (ms)", [fmt_ms(float(row["step_wall_p50_s"])) for row in rows]),
        metric_row("P95 (ms)", [fmt_ms(float(row["step_wall_p95_s"])) for row in rows]),
        metric_row("稳态 samples/s", [fmt(float(row["steady_samples_s"]), 2) for row in rows]),
        metric_row("端到端 samples/s", [fmt(float(row["e2e_samples_s"]), 2) for row in rows]),
        metric_row("稳态 TPS", [fmt(float(row["steady_tps"]), 2) for row in rows]),
        metric_row(
            "HBM 全 rank 峰值",
            ["unavailable" if row["hbm_max_gb"] is None else f"{float(row['hbm_max_gb']):.2f} GB" for row in rows],
        ),
        metric_row("loss（末步）", [fmt(float(row["last_loss"]), 4) for row in rows]),
    ])

    lines.extend([
        "",
        "## 2. Insight Step Trace 关键路径",
        "",
        "下列数值是 Insight 捕获步的各 rank 中位数，单位为毫秒（ms）。`Communication(Not Overlapped)` "
        "位于关键路径；`Communication` 时间线可包含与计算重叠的部分，不能与 Computing、Free 重复相加。"
        "`HCCL total` 是 collective/P2P 算子耗时累计，也可能重叠。",
        "",
    ])
    lines.extend(table_header("指标"))
    lines.extend([
        metric_row("捕获 step", [str(int(float(row["profile_step"]))) for row in rows]),
        metric_row("Stage / Step Trace (ms)", [fmt_ms(float(row["stage_s_median"])) for row in rows]),
        metric_row("Computing (ms)", [fmt_ms(float(row["computing_s_median"])) for row in rows]),
        metric_row(
            "Computing 占 Stage",
            [f"{float(row['computing_s_median']) / float(row['stage_s_median']):.2%}" for row in rows],
        ),
        metric_row(
            "CommNotOvl (ms)",
            [fmt_ms(float(row["communication_not_overlapped_s_median"])) for row in rows],
        ),
        metric_row(
            "Communication 总覆盖 (ms)",
            [fmt_ms(float(row["communication_total_s_median"])) for row in rows],
        ),
        metric_row("Free (ms)", [fmt_ms(float(row["free_s_median"])) for row in rows]),
        metric_row("Preparing (ms)", [fmt_ms(float(row["preparing_s_median"])) for row in rows]),
        metric_row("Bubble (ms)", [fmt_ms(float(row["bubble_s_median"])) for row in rows]),
        metric_row("HCCL total (ms)", [fmt_ms(float(row["hccl_total_s_median"])) for row in rows]),
    ])

    ref_stage_ms = reference["stage_ms"]
    ref_compute_ms = reference["computing_ms"]
    ref_comm_ms = reference["communication_not_overlapped_ms"]
    ref_free_ms = reference["free_ms"]
    ref_preparing_ms = reference["preparing_ms"]
    ref_main_sum_ms = ref_compute_ms + ref_comm_ms + ref_free_ms
    ref_compute_pct = ref_compute_ms / ref_stage_ms
    lines.extend([
        "",
        "`Stage` 是 Insight 中该次捕获的端到端单步耗时；它是一步的 Step Trace，"
        "不是 50 步平均，也不是从提交任务到退出的墙钟。",
        "",
        "| 指标 | 同事历史单步 |",
        "|---|---:|",
        f"| Stage (ms) | {ref_stage_ms:.0f} |",
        f"| Computing (ms) | {ref_compute_ms:.0f} |",
        f"| Computing 占 Stage | {ref_compute_pct:.2%} |",
        f"| CommNotOvl (ms) | {ref_comm_ms:.0f} |",
        f"| Free (ms) | {ref_free_ms:.0f} |",
        f"| Preparing (ms) | {ref_preparing_ms:.0f} |",
        f"| 主分项求和 (ms) | {ref_main_sum_ms:.0f} |",
        "",
        f"该记录满足 `{ref_compute_ms:.0f} + {ref_comm_ms:.0f} + {ref_free_ms:.0f} = "
        f"{ref_main_sum_ms:.0f} ms`，与 Stage `{ref_stage_ms:.0f} ms` 完全一致，所以 "
        f"**{ref_stage_ms:.0f} ms 就是这一采集步的 Step Trace 总耗时**。"
        "Preparing 只有 41 ms；若再加到主分项会变成 13,407 ms，说明它在该轨迹中是独立或重叠字段，"
        "不能再次相加。该记录只有耗时，没有模型、MBS、卡数、GC、attention 和代码版本，"
        "因此只能作为 Step Trace 量级参考，不能直接拿去比较 MFU 或判定性能优劣。",
        "",
        "## 3. MFU / HFU 三口径",
        "",
        "有效 MFU 使用实际训练工作量的分子；HFU 另外计入 checkpoint 重算。"
        "验收 60% 时应取有效 MFU，HFU 只说明设备实际执行了多少 FLOPs。",
        "",
    ])
    lines.extend(table_header("指标"))
    lines.extend([
        metric_row("文档一 MFU（486.354T）", [f"{float(row['document1_mfu']):.2%}" for row in rows]),
        metric_row("文档二 MFU（432T）", [f"{float(row['document2_mfu']):.2%}" for row in rows]),
        metric_row("文档二 HFU", [f"{float(row['document2_hfu']):.2%}" for row in rows]),
        metric_row(
            "MindSpeed backbone+ROSS MFU（486T）",
            [f"{float(row['mindspeed_plus_ross_mfu']):.2%}" for row in rows],
        ),
        metric_row(
            "MindSpeed HFU",
            [f"{float(row['mindspeed_plus_ross_hfu']):.2%}" for row in rows],
        ),
        metric_row(
            "修正 LM head MFU",
            [f"{float(row['mindspeed_plus_ross_valid-token_ce_mfu']):.2%}" for row in rows],
        ),
        metric_row(
            "修正 LM head HFU",
            [f"{float(row['mindspeed_plus_ross_valid-token_ce_hfu']):.2%}" for row in rows],
        ),
    ])

    lines.extend([
        "",
        "## 4. 扩展线性度",
        "",
        "线性度按固定 MBS 计算：2 台理想为单机吞吐的 2 倍，4 台理想为 4 倍。"
        "稳态和端到端分别计算，避免数据加载/启动开销掩盖不同结论。",
        "",
    ])
    scaling_columns = []
    for row in rows[1:]:
        node_multiplier = int(row["world_size"]) / int(single["world_size"])
        steady_speedup = float(row["steady_samples_s"]) / float(single["steady_samples_s"])
        e2e_speedup = float(row["e2e_samples_s"]) / float(single["e2e_samples_s"])
        step_ratio = float(row["step_wall_mean_s"]) / float(single["step_wall_mean_s"])
        scaling_columns.append((
            f"{scale_names.get(str(single['scale']), single['scale'])} → "
            f"{scale_names.get(str(row['scale']), row['scale'])}",
            steady_speedup,
            steady_speedup / node_multiplier,
            e2e_speedup,
            e2e_speedup / node_multiplier,
            step_ratio,
        ))

    if len(rows) >= 3:
        dual, quad = rows[1], rows[2]
        steady_speedup = float(quad["steady_samples_s"]) / float(dual["steady_samples_s"])
        e2e_speedup = float(quad["e2e_samples_s"]) / float(dual["e2e_samples_s"])
        scaling_columns.append((
            f"{scale_names.get(str(dual['scale']), dual['scale'])} → "
            f"{scale_names.get(str(quad['scale']), quad['scale'])}",
            steady_speedup,
            steady_speedup / 2,
            e2e_speedup,
            e2e_speedup / 2,
            float(quad["step_wall_mean_s"]) / float(dual["step_wall_mean_s"]),
        ))

    lines.extend([
        "| 指标 | " + " | ".join(column[0] for column in scaling_columns) + " |",
        "|---" + "|---:" * len(scaling_columns) + "|",
        metric_row("稳态 samples/s 倍数", [f"{column[1]:.3f}×" for column in scaling_columns]),
        metric_row("稳态线性度", [f"{column[2]:.2%}" for column in scaling_columns]),
        metric_row("端到端 samples/s 倍数", [f"{column[3]:.3f}×" for column in scaling_columns]),
        metric_row("端到端线性度", [f"{column[4]:.2%}" for column in scaling_columns]),
        metric_row("Step wall 比率", [f"{column[5]:.3f}×" for column in scaling_columns]),
    ])

    lines.extend([
        "",
        "## 5. 判读规则",
        "",
        "- 正式速度取 measure 轮；profile 会改变调度和采样，不能替代正式 TPS/MFU。",
        "- `time_elapsed` 是 50 步 Trainer runtime；Step wall 是稳态每步取最慢 rank 后再平均；报告中的耗时统一为 ms。",
        "- HBM 取所有 rank 的框架分配峰值，不是 rank 0 单卡值。",
        "- `CommNotOvl`、Communication 时间线和 HCCL 累计各自口径不同；只有 Stage 分解适合加总解释关键路径。",
        "- 若步时、loss、输入形状或 HBM 异常，本轮只用于诊断，不进入最终结论。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="append", type=parse_scale, required=True)
    parser.add_argument("--audit", action="append", type=parse_scale, required=True)
    parser.add_argument("--profile", action="append", type=parse_scale, required=True)
    parser.add_argument("--ross-costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    measures = dict(args.measure)
    audits = dict(args.audit)
    profiles = dict(args.profile)
    if not (measures.keys() == audits.keys() == profiles.keys()):
        parser.error("--measure, --audit and --profile must use identical scale names")
    if len(measures) < 1:
        parser.error("at least one scale is required")

    rows = [
        build_scale(name, measures[name], audits[name], profiles[name], args.ross_costs)
        for name in measures
    ]
    if [int(row["world_size"]) for row in rows] != sorted(int(row["world_size"]) for row in rows):
        raise SystemExit("scale arguments must be ordered from the smallest to largest world size")

    reference = {
        "stage_ms": 13366.0,
        "computing_ms": 12761.0,
        "communication_not_overlapped_ms": 242.0,
        "free_ms": 363.0,
        "preparing_ms": 41.0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    markdown = render(rows, reference)
    (args.output / "comparison.md").write_text(markdown + "\n", encoding="utf-8")

    millisecond_fields = {
        "runtime_ms": "runtime_s",
        "step_wall_mean_ms": "step_wall_mean_s",
        "step_wall_p50_ms": "step_wall_p50_s",
        "step_wall_p95_ms": "step_wall_p95_s",
        "stage_ms_median": "stage_s_median",
        "computing_ms_median": "computing_s_median",
        "communication_not_overlapped_ms_median": "communication_not_overlapped_s_median",
        "communication_total_ms_median": "communication_total_s_median",
        "overlapped_communication_ms_median": "overlapped_communication_s_median",
        "free_ms_median": "free_s_median",
        "preparing_ms_median": "preparing_s_median",
        "bubble_ms_median": "bubble_s_median",
        "hccl_total_ms_median": "hccl_total_s_median",
    }
    output_rows = []
    for row in rows:
        output_row = dict(row)
        output_row["node_list_display"] = row.get("node_list") or row.get("node") or ""
        for target, source in millisecond_fields.items():
            output_row[target] = float(row[source]) * 1000
        output_row["computing_stage_pct"] = (
            float(row["computing_s_median"]) / float(row["stage_s_median"]) * 100
        )
        output_rows.append(output_row)

    scale_names = {"single": "单机", "dual": "双机", "quad": "四机"}
    csv_columns = [
        f"{scale_names.get(str(row['scale']), str(row['scale']))}{row['world_size']}卡"
        for row in output_rows
    ]
    csv_metrics = [
        ("卡数", "world_size", str),
        ("节点", "node_list_display", str),
        ("MBS", "mbs", str),
        ("GBS", "gbs", str),
        ("time_elapsed_ms", "runtime_ms", lambda value: f"{float(value):.1f}"),
        ("step_wall_mean_ms", "step_wall_mean_ms", lambda value: f"{float(value):.1f}"),
        ("step_wall_p50_ms", "step_wall_p50_ms", lambda value: f"{float(value):.1f}"),
        ("step_wall_p95_ms", "step_wall_p95_ms", lambda value: f"{float(value):.1f}"),
        ("稳态_samples_per_s", "steady_samples_s", lambda value: f"{float(value):.2f}"),
        ("端到端_samples_per_s", "e2e_samples_s", lambda value: f"{float(value):.2f}"),
        ("稳态_TPS", "steady_tps", lambda value: f"{float(value):.2f}"),
        ("端到端_TPS", "e2e_tps", lambda value: f"{float(value):.2f}"),
        ("HBM全rank峰值_GB", "hbm_max_gb", lambda value: "" if value is None else f"{float(value):.2f}"),
        ("末步_loss", "last_loss", lambda value: f"{float(value):.4f}"),
        ("文档一_MFU", "document1_mfu", lambda value: f"{float(value):.2%}"),
        ("文档二_MFU", "document2_mfu", lambda value: f"{float(value):.2%}"),
        ("文档二_HFU", "document2_hfu", lambda value: f"{float(value):.2%}"),
        ("MindSpeed_backbone_plus_ROSS_MFU", "mindspeed_plus_ross_mfu", lambda value: f"{float(value):.2%}"),
        ("MindSpeed_HFU", "mindspeed_plus_ross_hfu", lambda value: f"{float(value):.2%}"),
        ("修正_LM_head_MFU", "mindspeed_plus_ross_valid-token_ce_mfu", lambda value: f"{float(value):.2%}"),
        ("修正_LM_head_HFU", "mindspeed_plus_ross_valid-token_ce_hfu", lambda value: f"{float(value):.2%}"),
        ("捕获_step", "profile_step", lambda value: str(int(float(value)))),
        ("Stage_ms", "stage_ms_median", lambda value: f"{float(value):.1f}"),
        ("Computing_ms", "computing_ms_median", lambda value: f"{float(value):.1f}"),
        ("Computing占Stage", "computing_stage_pct", lambda value: f"{float(value):.2f}%"),
        ("CommNotOvl_ms", "communication_not_overlapped_ms_median", lambda value: f"{float(value):.1f}"),
        ("Communication总覆盖_ms", "communication_total_ms_median", lambda value: f"{float(value):.1f}"),
        ("Free_ms", "free_ms_median", lambda value: f"{float(value):.1f}"),
        ("Preparing_ms", "preparing_ms_median", lambda value: f"{float(value):.1f}"),
        ("Bubble_ms", "bubble_ms_median", lambda value: f"{float(value):.1f}"),
        ("HCCL_total_ms", "hccl_total_ms_median", lambda value: f"{float(value):.1f}"),
    ]
    fieldnames = ["指标", *csv_columns]
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, key, formatter in csv_metrics:
            csv_row = {"指标": label}
            for column, output_row in zip(csv_columns, output_rows):
                csv_row[column] = formatter(output_row[key])
            writer.writerow(csv_row)
    (args.output / "summary.json").write_text(
        json.dumps({"scales": output_rows, "reference": reference}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(markdown)


if __name__ == "__main__":
    main()
