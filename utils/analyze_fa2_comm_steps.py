"""Inspect HCCL calls for selected raw Profiler step IDs and global ranks."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_rank_step(value: str) -> tuple[int, int]:
    rank, step = value.split(":", 1)
    return int(rank), int(step)


def find_rank_databases(root: Path) -> dict[int, Path]:
    databases: dict[int, Path] = {}
    for profile_dir in root.rglob("*_ascend_pt"):
        info_files = sorted(profile_dir.glob("profiler_info*.json"))
        if not info_files:
            continue
        with info_files[0].open(encoding="utf-8") as handle:
            rank = int(json.load(handle)["rank_id"])
        database = profile_dir / "ASCEND_PROFILER_OUTPUT" / "analysis.db"
        if not database.is_file():
            raise ValueError(f"Rank {rank} has no analysis database: {database}")
        if rank in databases:
            raise ValueError(f"Duplicate rank {rank}")
        databases[rank] = database
    return databases


def inspect_step(database: Path, rank: int, raw_step: int, top: int) -> None:
    step_name = f"step{raw_step}"
    with sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            """
            SELECT hccl_op_name, elapse_time, wait_time, transit_time,
                   synchronization_time, idle_time
            FROM CommAnalyzerTime
            WHERE step = ? AND lower(hccl_op_name) LIKE 'hcom_%'
            ORDER BY elapse_time DESC
            """,
            (step_name,),
        ).fetchall()
        families = connection.execute(
            """
            SELECT substr(hccl_op_name, 1, instr(hccl_op_name, '_AicpuKernel_') - 1)
                   AS family,
                   SUM(elapse_time), COUNT(*), SUM(wait_time), SUM(transit_time),
                   SUM(synchronization_time), SUM(idle_time)
            FROM CommAnalyzerTime
            WHERE step = ? AND lower(hccl_op_name) LIKE 'hcom_%'
            GROUP BY family
            ORDER BY SUM(elapse_time) DESC
            """,
            (step_name,),
        ).fetchall()

    print(f"\n## Rank {rank}, raw step {raw_step} ({len(rows)} HCCL calls)")
    print("\nOperation families:")
    print(
        "| Family | Elapsed ms | Calls | Wait ms | Transit ms | "
        "Sync ms | Idle ms |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, elapsed, calls, wait, transit, sync, idle in families:
        print(
            f"| `{name}` | {elapsed:.3f} | {calls} | {wait:.3f} | "
            f"{transit:.3f} | {sync:.3f} | {idle:.3f} |"
        )

    print(f"\nSlowest individual calls (top {top}):")
    print(
        "| Operation | Elapsed ms | Wait ms | Transit ms | Sync ms | Idle ms |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for name, elapsed, wait, transit, sync, idle in rows[:top]:
        print(
            f"| `{name}` | {elapsed:.3f} | {wait:.3f} | {transit:.3f} | "
            f"{sync:.3f} | {idle:.3f} |"
        )


def inspect_across_step(databases: dict[int, Path], raw_step: int) -> None:
    """Compare the same raw step across all ranks to expose a straggler."""
    step_name = f"step{raw_step}"
    rows: list[tuple[int, float, float, float, str, float, int]] = []
    for rank, database in sorted(databases.items()):
        with sqlite3.connect(
            database.resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            elapsed, calls, wait, transit = connection.execute(
                """
                SELECT SUM(elapse_time), COUNT(*), SUM(wait_time),
                       SUM(transit_time)
                FROM CommAnalyzerTime
                WHERE step = ? AND lower(hccl_op_name) LIKE 'hcom_allreduce%'
                """,
                (step_name,),
            ).fetchone()
            top_name, top_elapsed, top_wait = connection.execute(
                """
                SELECT hccl_op_name, elapse_time, wait_time
                FROM CommAnalyzerTime
                WHERE step = ? AND lower(hccl_op_name) LIKE 'hcom_allreduce%'
                ORDER BY elapse_time DESC LIMIT 1
                """,
                (step_name,),
            ).fetchone()
            if not calls:
                raise ValueError(f"Rank {rank} has no AllReduce data for {step_name}")
            if top_name is None:
                raise ValueError(f"Rank {rank} has aggregate data but no AllReduce row")
            rows.append(
                (rank, elapsed, wait, transit, top_name, top_elapsed, calls)
            )

    print(f"\n## AllReduce across all ranks, raw step {raw_step}")
    print(
        "| Rank | Elapsed ms | Calls | Wait ms | Transit ms | "
        "Slowest call | Slowest ms |"
    )
    print("|---:|---:|---:|---:|---:|---|---:|")
    for rank, elapsed, wait, transit, name, slowest, calls in rows:
        print(
            f"| {rank} | {elapsed:.3f} | {calls} | {wait:.3f} | "
            f"{transit:.3f} | `{name}` | {slowest:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--rank-step",
        type=parse_rank_step,
        action="append",
        metavar="RANK:STEP",
        help="Global rank and raw STEP_TIME ID; repeat for multiple cases.",
    )
    parser.add_argument(
        "--across-step",
        type=int,
        action="append",
        metavar="STEP",
        help="Compare AllReduce totals for one raw STEP_TIME ID across all ranks.",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    if not args.rank_step and not args.across_step:
        parser.error("Provide --rank-step and/or --across-step")

    databases = find_rank_databases(args.root)
    for rank, raw_step in args.rank_step or []:
        if rank not in databases:
            raise ValueError(f"Rank {rank} not found")
        inspect_step(databases[rank], rank, raw_step, args.top)
    for raw_step in args.across_step or []:
        inspect_across_step(databases, raw_step)


if __name__ == "__main__":
    main()
