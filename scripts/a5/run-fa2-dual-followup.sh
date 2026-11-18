#!/usr/bin/env bash
# Continue after the original clean 50-step run without relaunching it.
set -euo pipefail
case "$(hostname -s)" in
  liteserver-a5-00003) node=a53; peer=a54 ;;
  liteserver-a5-00004) node=a54; peer=a53 ;;
  *) exit 2 ;;
esac
[[ $# == 1 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$ ]] || exit 2
id=$1
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
control=$root/$id
out=$root/mfu-a53-a54-$id
[[ -d "$out-measure" && -d "$control" ]] || exit 2
status() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$control/$node.state"; }
trap 'rc=$?; if [[ $rc != 0 ]]; then touch "$control/$node.failed"; status "stopped (exit $rc)"; fi' EXIT
wait_for() {
  local f=$1 deadline=$((SECONDS+$2))
  until [[ -s "$f" ]]; do
    [[ ! -f "$control/$peer.failed" && $SECONDS -lt $deadline ]] || return 1
    sleep 8
  done
}
status "waiting for clean measure on both nodes"
for n in a53 a54; do
  wait_for "$out-measure/train-$n.exit" 1800
  [[ "$(< "$out-measure/train-$n.exit")" == 0 ]]
done
wait_for "$out-measure/summary.txt" 300
summary_wait_start=$SECONDS
until grep -q '^fast:' "$out-measure/summary.txt"; do
  [[ ${SECONDS} -lt $((summary_wait_start+300)) ]] || exit 1
  sleep 2
done
export ZB_PAIR_RUN="$id" ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10
export ZB_FA2_UNPAD_CACHE=1 ZB_FA2_UNPAD_STATS=1 ZB_FA2_UNPAD_MARKERS=0
unset ZB_SINGLE_RUN
for job in audit profile; do
  status "$job: waiting for both nodes to be idle"
  deadline=$((SECONDS+1800))
  while :; do
    [[ ! -f "$control/$peer.failed" && $SECONDS -lt $deadline ]] || exit 1
    stats=$(npu-smi info)
    if awk '/Ascend950DT/{c++} /No running processes found in NPU [0-7]/{i++}
            END{exit !(c==8 && i==8)}' <<< "$stats"; then break; fi
    sleep 15
  done
  printf 'ready\n' > "$control/$node.$job.ready"
  wait_for "$control/$peer.$job.ready" 1850
  status "$job running"
  if [[ "$job" == profile ]]; then
    ZB_PROFILE_FULL=1 ZB_FA2_UNPAD_MARKERS=1 ZB_JOB=profile bash "$here/benchmark-mfu.sh"
  else
    ZB_JOB=audit bash "$here/../tri-mfu-1112-fa2-v1/benchmark-mfu.sh"
  fi
  wait_for "$out-$job/train-$peer.exit" 3600
  [[ "$(< "$out-$job/train-$peer.exit")" == 0 ]]
  if [[ "$job" == audit && "$node" == a53 ]]; then
    box_code=/work/zhaobo/artifacts/code/${here##*/}/checkout
    docker exec zhaobo-drivevla-a53 bash --noprofile --norc -e -c '
      source /work/zhaobo/bootstrap/enter-env.sh
      cd "$1"
      root=/work/zhaobo/artifacts/runs/mfu
      exec env DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python utils/report_projection_mfu.py \
        "$root/mfu-a53-a54-$2-measure" "$root/mfu-a53-a54-$2-audit" --ross-costs "$root/zhao-mfu01/ross.json"
    ' _ "$box_code" "$id" > "$control/optimized-comparison.tmp"
    mv "$control/optimized-comparison.tmp" "$control/optimized-comparison.md"
  fi
done
status completed
