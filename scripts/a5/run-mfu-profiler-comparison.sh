#!/usr/bin/env bash
# Collect comparable 8/16-card Insight traces separately from clean MFU timing.
set -euo pipefail
[[ $# -ge 1 && $# -le 2 && "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || exit 2
id=$1
single_id=${2:-$id}
[[ "$single_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || exit 2
case "$(hostname -s)" in
  liteserver-a5-00003) node=a53; peer=a54 ;;
  liteserver-a5-00004) node=a54; peer=a53 ;;
  *) exit 2 ;;
esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
control=$root/three-methods-tri1112v1/$id
[[ -d "$control" ]] || exit 2
export ZB_JOB=profile ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=32 ZB_MFU_WARMUP=10
unset ZB_SINGLE_RUN ZB_PAIR_RUN

status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee "$control/$node.state"
}
finish() {
  rc=$?
  if [[ $rc != 0 ]]; then
    touch "$control/$node.failed"
    status "stopped (exit $rc)"
  fi
}
trap finish EXIT
check_peer() {
  [[ ! -e "$control/$peer.failed" ]]
}
wait_for() {
  local deadline=$((SECONDS+$2))
  until [[ -e "$control/$1" ]]; do
    check_peer && [[ $SECONDS -lt $deadline ]] || return 1
    sleep 10
  done
}
wait_idle() {
  local deadline=$((SECONDS+${1:-1800})) stats stable=0
  while check_peer && [[ $SECONDS -lt $deadline ]]; do
    stats=$(npu-smi info) || return 1
    if awk '/Ascend950DT/{cards++; if ($0 !~ /\| OK +\|/) bad++}
            /No running processes found in NPU [0-7]/{idle[$(NF-1)]=1}
            END{for(i in idle)n++; exit !(cards==8 && n==8 && !bad)}' <<< "$stats"; then
      [[ "$stable" == 0 ]] || return 0
      stable=1
      sleep 8
    else
      stable=0
      sleep 30
    fi
  done
  return 1
}

single=$root/mfu-a53-$single_id-profile
dual=$root/mfu-a53-a54-$id-profile
if [[ "$node" == a53 ]]; then
  if [[ "$single_id" == "$id" ]]; then
    status "single profile: waiting for 8 idle NPUs"
    wait_idle
    status "single profile: running"
    ZB_SINGLE_RUN="$id" bash "$here/benchmark-mfu.sh"
  else
    status "reusing successful single profile: $single_id"
    [[ "$(< "$single/train-a53.exit")" == 0 ]] || exit 1
  fi
  python3 "$here/report_mfu_profiler.py" "$single" | tee "$control/single-profile.md"
  touch "$control/single.done"
else
  status "waiting for the single profile"
  wait_for single.done 4200
fi

status "dual profile: waiting for 8 idle NPUs"
wait_idle
touch "$control/$node.ready"
wait_for "$peer.ready" 1860
wait_idle 90
status "dual profile: running"
ZB_PAIR_RUN="$id" bash "$here/benchmark-mfu.sh"
touch "$control/$node.done"
if [[ "$node" == a53 ]]; then
  wait_for "$peer.done" 1200
  python3 "$here/report_mfu_profiler.py" "$single" "$dual" | tee "$control/paired-profile.md"
  touch "$control/paired.done"
else
  wait_for paired.done 300
fi
status "completed"
