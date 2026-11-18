#!/usr/bin/env bash
# Run the 8-NPU pair of measurements first, then the matched 16-NPU pair.
set -euo pipefail
case "$(hostname -s)" in
  liteserver-a5-00003) node=a53 ;;
  liteserver-a5-00004) node=a54 ;;
  *) exit 2 ;;
esac
[[ $# == 1 && "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || exit 2
id=$1
shared=/mnt/sfs_turbo/zhaobo/drivevla-w0
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ "$here" == "$shared/"* && -d "$here/checkout" ]] || exit 2
root=$shared/artifacts/runs/mfu
control=$root/three-methods-$id
[[ -d "$control" ]] || exit 2
checkout=/work/zhaobo/${here#"$shared/"}/checkout
box=zhaobo-drivevla-$node

export ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10
unset ZB_SINGLE_RUN ZB_PAIR_RUN

status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee "$control/$node.state"
}
finish() {
  local rc=$?
  trap - EXIT
  if [[ "$rc" != 0 ]]; then
    touch "$control/$node.failed"
    status "stopped (exit $rc)"
  fi
  exit "$rc"
}
trap finish EXIT
check_peer() {
  [[ ! -e "$control/a53.failed" && ! -e "$control/a54.failed" ]]
}
wait_marker() {
  local marker=$1 deadline=$((SECONDS + $2))
  until [[ -e "$control/$marker" ]]; do
    check_peer || return 1
    [[ $SECONDS -lt $deadline ]] || { echo "Timed out waiting for $marker"; return 1; }
    sleep 10
  done
}
wait_idle() {
  local deadline=$((SECONDS + 1800)) stats
  while :; do
    check_peer || return 1
    stats=$(npu-smi info) || return 1
    if awk '/Ascend950DT/{cards++; if ($0 !~ /\| OK +\|/) bad++}
            /No running processes found in NPU [0-7]/{idle[$(NF-1)]=1}
            END{for (i in idle) n++; exit !(cards==8 && n==8 && !bad)}' <<< "$stats"; then
      return 0
    fi
    [[ $SECONDS -lt $deadline ]] || { echo 'Timed out waiting for eight idle NPUs'; return 1; }
    sleep 30
  done
}
compare() {
  local kind=$1 prefix="mfu-$2-$id"
  local out=/work/zhaobo/artifacts/runs/mfu
  local args=("$out/$prefix-measure" "$out/$prefix-audit" --ross-costs "$out/zhao-mfu01/ross.json")
  if [[ "$kind" == dual ]]; then
    args+=(--single "$out/mfu-a53-$id-measure")
  fi
  docker exec "$box" bash --noprofile --norc -e -c '
    source /work/zhaobo/bootstrap/enter-env.sh
    cd "$1"
    shift
    exec env DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python utils/report_projection_mfu.py "$@"
  ' _ "$checkout" "${args[@]}" | tee "$control/$kind-comparison.md"
}

if [[ "$node" == a53 ]]; then
  for job in measure audit; do
    status "single $job: waiting for idle NPUs"
    wait_idle
    status "single $job: running"
    ZB_SINGLE_RUN="$id" ZB_JOB="$job" bash "$here/benchmark-mfu.sh"
  done
  compare single a53
  touch "$control/single.done"
else
  status "waiting for the A53 single-node report"
  wait_marker single.done 4800
fi

peer=a53
[[ "$node" != a53 ]] || peer=a54
for job in measure audit; do
  status "dual $job: waiting for idle NPUs on both nodes"
  wait_idle
  touch "$control/$node.$job.ready"
  wait_marker "$peer.$job.ready" 1860
  check_peer
  status "dual $job: running"
  ZB_PAIR_RUN="$id" ZB_JOB="$job" bash "$here/benchmark-mfu.sh"
  touch "$control/$node.$job.done"
  wait_marker "$peer.$job.done" 1500
done
if [[ "$node" == a53 ]]; then
  compare dual a53-a54
  touch "$control/dual.done"
else
  wait_marker dual.done 300
fi
status "completed"
