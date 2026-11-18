#!/usr/bin/env bash
# Two short clean A53 trials; toggle only the unpadding cache.
set -euo pipefail
[[ $(hostname -s) == liteserver-a5-00003 && $# == 1 && "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,45}$ ]] || exit 2
id=$1
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
control=$root/$id
[[ -d "$control" ]] || exit 2
unset ZB_PAIR_RUN
export ZB_JOB=measure ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=12 ZB_MFU_WARMUP=5
export ZB_FA2_UNPAD_STATS=1 ZB_FA2_UNPAD_MARKERS=0
status() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$control/state"; }
trap 'rc=$?; if [[ $rc != 0 ]]; then status "stopped (exit $rc)"; fi' EXIT
status "baseline running (cache off)"
ZB_SINGLE_RUN="$id-base" ZB_FA2_UNPAD_CACHE=0 bash "$here/benchmark-mfu.sh"
status "experiment running (cache on)"
ZB_SINGLE_RUN="$id-cache" ZB_FA2_UNPAD_CACHE=1 bash "$here/benchmark-mfu.sh"
python3 "$here/checkout/utils/report_fa2_ab.py" \
  "$root/mfu-a53-$id-base-measure" "$root/mfu-a53-$id-cache-measure" | tee "$control/comparison.md"
status completed
