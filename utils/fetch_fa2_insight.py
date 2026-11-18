"""Download 50-step Insight data for fa2dual1113-01.

``--analysis-only`` is enough for MindStudio Insight cluster timelines and
communication analysis, while keeping the transfer small enough for routine
review. The default mode downloads the full raw operator databases as well.
"""

import argparse
import json
import sqlite3
import subprocess
import tarfile
import time
from pathlib import Path


def fetch(destination, wait_seconds=0, analysis_only=False):
    remote = "/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/mfu-a53-a54-fa2dual1113-01-profile"
    ctrl = "/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/fa2dual1113-01"
    deadline = time.monotonic() + wait_seconds
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "ott9-a53"]
    guard = (
        f"if test -e '{ctrl}/a53.failed' || test -e '{ctrl}/a54.failed'; then exit 3; fi; "
        f"test -s '{remote}/train-a53.exit' && test -s '{remote}/train-a54.exit' && "
        f"grep -qx 0 '{remote}/train-a53.exit' && grep -qx 0 '{remote}/train-a54.exit'"
    )
    while True:
        rc = subprocess.run([*ssh, guard]).returncode
        if rc == 0:
            break
        if rc != 1 or time.monotonic() >= deadline:
            raise RuntimeError(f"Profile did not complete (status {rc})")
        time.sleep(30)
    names = ["analysis.db", "profiler_info*.json", "profiler_metadata.json"]
    if not analysis_only:
        names.insert(0, "ascend_pytorch_profiler_*.db")
    name_args = " -o ".join(f"-name '{name}'" for name in names)
    command = (
        f"set -o pipefail; cd '{remote}' && grep -qx 0 train-a53.exit && grep -qx 0 train-a54.exit && "
        f"(find . -type f \\( {name_args} \\) "
        "-print0 | tar --null -T - -cf - | pigz -1 -p 4)"
    )
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / ("import-analysis-db" if analysis_only else "import-db")
    if target.exists():
        raise FileExistsError(target)
    staging = destination / (target.name + ".partial")
    staging.mkdir()
    # Extract directly into the shallow layout accepted by Insight, without
    # keeping a duplicate compressed archive or loading entire databases in RAM.
    with subprocess.Popen([*ssh, command], stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as archive:
            for member in archive:
                parts = Path(member.name).parts
                idx = next((i for i, p in enumerate(parts) if p.endswith("_ascend_pt")), None)
                if idx is None:
                    continue
                if not member.isfile():
                    raise ValueError(f"Unexpected archive member: {member.name}")
                member.name = "/".join(parts[idx:])
                archive.extract(member, staging, filter="data")
        if proc.wait():
            raise RuntimeError("SSH export failed")
    ranks = {}
    profile_dirs = sorted({
        path.parent
        for path in staging.rglob("profiler_info*.json")
    })
    for profile_dir in profile_dirs:
        info_files = sorted(profile_dir.glob("profiler_info*.json"))
        if len(info_files) != 1:
            raise ValueError(f"Expected one profiler_info file under {profile_dir}")
        with info_files[0].open(encoding="utf-8") as handle:
            rank = int(json.load(handle)["rank_id"])
        if rank in ranks:
            raise ValueError(f"Duplicate profiler rank {rank}")
        analysis_path = profile_dir / "ASCEND_PROFILER_OUTPUT" / "analysis.db"
        if not analysis_path.is_file():
            raise ValueError(f"Rank {rank} missing analysis.db")
        with sqlite3.connect(
            analysis_path.resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            steps = [row[0] for row in db.execute("SELECT step FROM StepTraceTime")]
        if len(steps) != 50 or len(set(steps)) != 50:
            raise ValueError(f"Rank {rank} has unexpected analysis steps: {steps}")
        ranks[rank] = len(steps)
    if sorted(ranks) != list(range(16)):
        raise ValueError(f"Expected 16 ranks; found {sorted(ranks)}")
    # On Windows, a scanner can briefly retain a handle after the final SQLite
    # validation closes. Retry the directory rename instead of discarding a
    # successfully downloaded and validated package.
    for attempt in range(10):
        try:
            staging.rename(target)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(2)
    print(f"Ranks and captured step counts: {ranks}")
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    p.add_argument("--wait-seconds", type=int, default=0)
    p.add_argument(
        "--analysis-only",
        action="store_true",
        help="download MindStudio communication/timeline DBs without the ~3 GiB/rank raw operator DB",
    )
    a = p.parse_args()
    fetch(a.destination, a.wait_seconds, a.analysis_only)
