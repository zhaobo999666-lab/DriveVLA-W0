#!/usr/bin/env bash
# Wait for a stable idle window on A53, then run one 8-NPU performance baseline.
set -euo pipefail

[[ $(hostname -s) == liteserver-a5-00003 ]] || {
  printf 'This launcher is restricted to A53.\n' >&2; exit 2;
}
[[ $# == 1 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,48}$ ]] || {
  printf 'Usage: %s RUN_ID\n' "$0" >&2; exit 2;
}

id=$1
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
control=$root/wait-single-$id
mkdir "$control"

interval=${ZB_WAIT_INTERVAL:-15}
required=${ZB_WAIT_STABLE_CHECKS:-4}
[[ "$interval" =~ ^[1-9][0-9]*$ && "$required" =~ ^[1-9][0-9]*$ ]] || exit 2

status() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$control/state"; }
finish() {
  rc=$?
  printf '%s\n' "$rc" > "$control/exit"
  if [[ $rc != 0 ]]; then status "stopped (exit $rc)"; fi
}
trap finish EXIT

stable=0
status 'waiting for eight idle NPUs and free ports 29555/29556'
while (( stable < required )); do
  stats=$(npu-smi info) || { stable=0; sleep "$interval"; continue; }
  ports=$(ss -H -lnt 'sport = :29555 or sport = :29556') || {
    stable=0; sleep "$interval"; continue;
  }
  if [[ -z "$ports" ]] && awk '
    /Ascend950DT/ { cards++ }
    /No running processes found in NPU [0-7][[:space:]]*\|/ { idle++ }
    END { exit !(cards == 8 && idle == 8) }
  ' <<< "$stats"; then
    ((++stable))
    status "idle check $stable/$required"
  else
    if (( stable > 0 )); then status 'idle window interrupted; restarting checks'; fi
    stable=0
  fi
  (( stable >= required )) || sleep "$interval"
done

status 'stable idle window confirmed; starting 8-NPU 50-step baseline'
export ZB_SINGLE_RUN="$id" ZB_JOB=measure ZB_MFU_PROFILE=fast ZB_MFU_SMOKE=0
export ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10
export ZB_FA2_UNPAD_CACHE=1 ZB_FA2_UNPAD_MARKERS=0 ZB_FA2_UNPAD_STATS=0
export ZB_ZERO2_BUCKET_SIZE=auto ZB_ZERO2_OVERLAP_COMM=1 ZB_HCCL_DETERMINISTIC=1
bash "$here/benchmark-mfu.sh" |& tee "$control/launcher.log"
status completed
