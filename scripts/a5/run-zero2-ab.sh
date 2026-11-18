#!/usr/bin/env bash
# Run the same sequential ZeRO-2 A/B matrix on A53 and A54.
set -euo pipefail

case "$(hostname -s)" in
  liteserver-a5-00003) node=a53; peer=a54 ;;
  liteserver-a5-00004) node=a54; peer=a53 ;;
  *) printf 'Run on A53 or A54.\n' >&2; exit 2 ;;
esac
[[ $# == 1 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,38}$ ]] || {
  printf 'Usage: %s RUN_ID\n' "$0" >&2; exit 2;
}

id=$1
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
control=$root/zero2-ab-$id
if [[ "$node" == a53 ]]; then
  mkdir "$control"
else
  for _ in $(seq 1 90); do [[ -d "$control" ]] && break; sleep 2; done
  [[ -d "$control" ]] || { printf 'A53 did not create %s\n' "$control" >&2; exit 1; }
fi

status() { printf '%s %s %s\n' "$(date -Is)" "$node" "$*" | tee "$control/$node.state"; }
finish() {
  rc=$?
  if [[ $rc != 0 ]]; then
    touch "$control/$node.failed"
    status "stopped (exit $rc)"
  fi
}
trap finish EXIT

export ZB_JOB=measure ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=${ZB_AB_STEPS:-30} ZB_MFU_WARMUP=${ZB_AB_WARMUP:-10}
export ZB_FA2_UNPAD_CACHE=1 ZB_FA2_UNPAD_MARKERS=0 ZB_FA2_UNPAD_STATS=0
unset ZB_SINGLE_RUN

variants=(base bucket2x bucket4x detoff overlapoff)
for variant in "${variants[@]}"; do
  export ZB_ZERO2_BUCKET_SIZE=auto ZB_ZERO2_OVERLAP_COMM=1 ZB_HCCL_DETERMINISTIC=1
  case "$variant" in
    base) ;;
    bucket2x) ZB_ZERO2_BUCKET_SIZE=33554432 ;;
    bucket4x) ZB_ZERO2_BUCKET_SIZE=67108864 ;;
    detoff) ZB_HCCL_DETERMINISTIC=0 ;;
    overlapoff) ZB_ZERO2_OVERLAP_COMM=0 ;;
  esac
  export ZB_ZERO2_BUCKET_SIZE ZB_ZERO2_OVERLAP_COMM ZB_HCCL_DETERMINISTIC
  status "$variant running"
  ZB_PAIR_RUN="$id-$variant" bash "$here/benchmark-mfu.sh"
  touch "$control/$node.$variant.done"
done

status 'waiting for both nodes'
for variant in "${variants[@]}"; do
  for _ in $(seq 1 180); do
    [[ -f "$control/$peer.$variant.done" ]] && break
    [[ ! -f "$control/$peer.failed" ]] || exit 1
    sleep 5
  done
  [[ -f "$control/$peer.$variant.done" ]] || exit 1
done

if [[ "$node" == a53 ]]; then
  runs=()
  for variant in "${variants[@]}"; do
    runs+=("$root/mfu-a53-a54-$id-$variant-measure")
  done
  python3 "$here/checkout/utils/report_zero2_ab.py" "${runs[@]}" \
    --output "$control/comparison.md" | tee "$control/comparison.stdout"
fi
status completed
