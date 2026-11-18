"""Summarize the critical-path and operator metrics from local Insight DBs."""

import argparse
from bisect import bisect_right
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median


def collect(path, *, single_step=None):
    rank = int(path.stem.rsplit("_", 1)[-1])
    with sqlite3.connect((path.parent/"analysis.db").resolve().as_uri()+"?mode=ro", uri=True) as analysis:
        summary = {int(s): values for s, *values in analysis.execute(
            """SELECT step,stage,computing,communication,communication_not_overlapped,free,bubble
               FROM StepTraceTime""") if str(s).isdigit()}
        hccl = {int(str(s).removeprefix("step")): elapsed for s, elapsed in analysis.execute(
            """SELECT step, SUM(elapse_time) FROM CommAnalyzerTime
               WHERE lower(hccl_op_name) LIKE 'hcom_%' GROUP BY step""")
            if str(s).removeprefix("step").isdigit()}
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as db:
        spans = db.execute("SELECT id,startNs,endNs FROM STEP_TIME ORDER BY id").fetchall()
        if not spans or len({r[0] for r in spans}) != len(spans):
            raise ValueError(f"Invalid step table: {path}")
        steps = [s for s, _, _ in spans]
        offset = 0 if min(steps) == 0 else 1
        rows = {}
        for s, lo, hi in spans:
            ordinal = s + 1 - offset
            if (single_step is not None and s != single_step) or (
                single_step is None and not 11 <= ordinal <= 50
            ):
                continue
            fields = ("stage_s", "compute_s", "comm_total_s", "comm_unoverlapped_s", "free_s", "bubble_s")
            if s not in summary or s not in hccl:
                raise ValueError(f"No Insight communication summary: {path}, step {s}")
            row = dict(zip(fields, (v/1e6 for v in summary[s])))
            row["hccl_elapsed_s"] = hccl[s]/1e3
            if not row.get("compute_s") or not row.get("comm_total_s"):
                raise ValueError(f"No compute/communication data: {path}, step {s}")
            if row.get("comm_unoverlapped_s", 0) > row["comm_total_s"] + .001:
                raise ValueError(f"Inconsistent overlap: {path}, step {s}")
            rows[ordinal] = row
        # ``rows`` is keyed by optimizer ordinal, while ``single_step`` is a raw
        # database STEP_TIME id. Convert after detecting the database offset.
        expected = (
            {single_step + 1 - offset}
            if single_step is not None
            else set(range(11, 51))
        )
        if not expected.issubset(rows):
            raise ValueError(f"Incomplete requested window: {path}, captured {steps}")
        selected = [x for x in spans if x[0]+1-offset in rows]
        lo, hi = selected[0][1], selected[-1][2]
        gaps = db.execute(
            """SELECT startNs,endNs FROM OVERLAP_ANALYSIS
               WHERE type=3 AND startNs<? AND endNs>? ORDER BY startNs""", (hi, lo)).fetchall()
        pos = 0
        for step, start, end in selected:
            while pos < len(gaps) and gaps[pos][1] <= start:
                pos += 1
            top, top_start = 0, start
            j = pos
            while j < len(gaps) and gaps[j][0] < end:
                left, right = gaps[j]
                a, b = max(left, start), min(right, end)
                if b-a > top:
                    top, top_start = b-a, a
                j += 1
            row = rows[step+1-offset]
            row["largest_free_s"] = top/1e9
            row["largest_free_offset_s"] = (top_start-start)/1e9
        ops = db.execute("""
            SELECT n.value, SUM(t.endNs-t.startNs), COUNT(*)
            FROM TASK t JOIN COMPUTE_TASK_INFO m USING(globalTaskId)
            JOIN STRING_IDS n ON n.id=m.opType
            WHERE t.startNs>=? AND t.endNs<=? GROUP BY n.value""", (lo, hi)).fetchall()
        matmul = db.execute("""
            SELECT COALESCE(i.value,''), SUM(t.endNs-t.startNs), COUNT(*)
            FROM TASK t JOIN COMPUTE_TASK_INFO m USING(globalTaskId)
            JOIN STRING_IDS n ON n.id=m.opType
            LEFT JOIN STRING_IDS i ON i.id=m.inputShapes
            WHERE n.value='MatMulV3' AND t.startNs>=? AND t.endNs<=? GROUP BY i.value
            """, (lo,hi)).fetchall()
        groups = defaultdict(lambda: [0, 0])
        for shape, dur, count in matmul:
            axes = set(shape.strip('"').replace(";",",").split(","))
            if "4096" in axes and axes & {"12288", "24576"}:
                kind = "Text MLP MatMulV3"
            elif "1152" in axes and "4304" in axes:
                kind = "ViT MLP MatMulV3"
            elif "1152" in axes and "3456" in axes:
                kind = "ViT QKV MatMulV3"
            else:
                continue
            groups[kind][0] += dur
            groups[kind][1] += count
        # A fused gate/up or down projection with its forward input dimensions
        # also appears in checkpoint replay. Attribute it using the correlated
        # launch on the host, not by assuming it executes within a host span.
        launches = db.execute("""
            SELECT t.globalTaskId, t.endNs-t.startNs, MIN(CAST(h.startNs AS INTEGER)), h.globalTid
            FROM TASK t JOIN COMPUTE_TASK_INFO m USING(globalTaskId)
            JOIN STRING_IDS n ON n.id=m.opType
            JOIN STRING_IDS i ON i.id=m.inputShapes
            LEFT JOIN CONNECTION_IDS link ON link.connectionId=t.connectionId
            LEFT JOIN PYTORCH_API h ON h.connectionId=link.id AND h.type=50001
            WHERE n.value='MatMulV3' AND t.startNs>=? AND t.endNs<=?
              AND (i.value LIKE '%4096;24576,4096"' OR i.value LIKE '%12288;4096,12288"')
            GROUP BY t.globalTaskId""", (lo, hi)).fetchall()
        scopes = defaultdict(list)
        for a, b, tid in db.execute("""
            SELECT CAST(p.startNs AS INTEGER), CAST(p.endNs AS INTEGER), p.globalTid
            FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
            WHERE n.value='CheckpointFunctionBackward' ORDER BY CAST(p.startNs AS INTEGER)"""):
            scopes[tid].append((a,b))
        starts = {tid:[a for a,_ in v] for tid,v in scopes.items()}
        key = "Text MLP replay MatMulV3"
        for _, duration, host_start, tid in launches:
            if host_start is None or tid not in starts:
                continue
            i = bisect_right(starts[tid], host_start)-1
            if i >= 0 and host_start <= scopes[tid][i][1]:
                groups[key][0] += duration
                groups[key][1] += 1
        labels = db.execute("""
            SELECT n.value, SUM(CAST(p.endNs AS INTEGER)-CAST(p.startNs AS INTEGER)),COUNT(*)
            FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
            WHERE n.value IN ('DriveVLA::FA2_unpad_metadata','aten::nonzero')
            AND CAST(p.startNs AS INTEGER)>=? AND CAST(p.endNs AS INTEGER)<=? GROUP BY n.value
            """, (lo, hi)).fetchall()
    scale = len(rows)
    return rank, steps, rows, {k: (d/1e9/scale, count/scale) for k,d,count in
                              ops + [(k,*v) for k,v in groups.items()]}, {
        k: (d/1e9/scale,count/scale) for k,d,count in labels
    }


def describe_gap(path, raw_step, offset_s, duration_s):
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as db:
        start, = db.execute("SELECT startNs FROM STEP_TIME WHERE id=?", (raw_step,)).fetchone()
        a = start + round(offset_s*1e9)
        b = a + round(duration_s*1e9)
        return db.execute("""
            SELECT n.value, SUM(MIN(CAST(p.endNs AS INTEGER),?)-MAX(CAST(p.startNs AS INTEGER),?))/1e9,
                   COUNT(*)
            FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
            WHERE CAST(p.startNs AS INTEGER)<? AND CAST(p.endNs AS INTEGER)>?
            GROUP BY n.value ORDER BY 2 DESC LIMIT 12
            """, (b,a,b,a)).fetchall()


def report(root, csv_path, single_step=None):
    paths = {}
    for p in root.rglob("ascend_pytorch_profiler_*.db"):
        rank = int(p.stem.rsplit("_", 1)[-1])
        if rank in paths:
            raise ValueError(f"Duplicate profiler database for rank {rank}")
        paths[rank] = p
    results = [collect(paths[r], single_step=single_step) for r in sorted(paths)]
    if sorted(r[0] for r in results) != list(range(16)):
        raise ValueError("Expected all 16 distinct ranks")
    if any(r[1] != results[0][1] for r in results):
        raise ValueError("Rank step IDs differ")
    fields = ("stage_s", "compute_s", "comm_total_s", "comm_unoverlapped_s", "free_s", "bubble_s",
              "hccl_elapsed_s", "largest_free_s", "largest_free_offset_s")
    pool = defaultdict(list)
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("rank","optimizer_step",*fields))
        for rank, _, steps, _, _ in results:
            for ordinal,row in steps.items():
                vals = [row.get(k,0) for k in fields]
                writer.writerow((rank,ordinal,*vals))
                for k,v in zip(fields,vals):
                    pool[k].append(v)
    n = len(results[0][2])
    print(f"16 ranks, database step IDs {results[0][1][0]}–{results[0][1][-1]}; {n} selected steps/rank\n")
    print("| Timeline (s/rank/step) | Median | Min | Max |\n|---|---:|---:|---:|")
    for k in fields:
        v=pool[k]
        print(f"| {k} | {median(v):.6f} | {min(v):.6f} | {max(v):.6f} |")
    for k in ("stage_s", "comm_unoverlapped_s", "largest_free_s"):
        seconds, rank, step = max((v.get(k, 0), r[0], step) for r in results for step, v in r[2].items())
        print(f"\nLargest {k}: {seconds:.6f} s at rank {rank}, optimizer step {step}.")
    top = results[rank][2][step]
    raw_step = step - int(results[rank][1][0] == 0)
    print(f"\nHost calls overlapping that gap, offset {top['largest_free_offset_s']:.6f} s "
          "(nested durations, not causal attribution):\n")
    for name, elapsed, calls in describe_gap(paths[rank], raw_step, top["largest_free_offset_s"], seconds):
        print(f"- {name}: {elapsed:.6f} s across {calls} calls")
    for title,index in (("Device operators",3),("Host markers / API ranges",4)):
        keys = set().union(*(r[index] for r in results))
        totals = {key: tuple(median(r[index].get(key,(0,0))[i] for r in results) for i in (0,1)) for key in keys}
        print(f"\n{title}: median of per-rank steady means; cumulative durations may overlap.\n")
        print("| Name | Accumulated s/step | Calls/step |\n|---|---:|---:|")
        for k,(t,c) in sorted(totals.items(), key=lambda item:item[1][0], reverse=True)[:15]:
            print(f"| {k} | {t:.6f} | {c:.2f} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--single-step", type=int, help="restrict to a raw database step ID")
    a=p.parse_args()
    report(a.root, a.csv, a.single_step)
