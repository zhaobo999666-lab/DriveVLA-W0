#!/usr/bin/env bash
# Coordinate matched 8-, 16-, and 32-NPU performance runs on A51-A54.
#
# Run the same command on A51-A54 with the same RUN_ID. A51 drives the single
# node baseline; A51/A52 then run the dual-node stage; all four nodes wait for
# an idle window and run the four-node stage. Coordination uses the shared disk.
set -euo pipefail

case "$(hostname -s)" in
  liteserver-a5-00001) node=a51; rank=0 ;;
  liteserver-a5-00002) node=a52; rank=1 ;;
  liteserver-a5-00003) node=a53; rank=2 ;;
  liteserver-a5-00004) node=a54; rank=3 ;;
  *) printf 'Run this launcher on A51, A52, A53 or A54.\n' >&2; exit 2 ;;
esac
[[ $# == 1 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$ ]] || {
  printf 'Usage: %s RUN_ID\n' "$0" >&2
  exit 2
}

id=$1
shared=/mnt/sfs_turbo/zhaobo/drivevla-w0
root=$shared/artifacts/runs/mfu
control=$root/scaling-$id
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ "$here" == "$shared/"* ]] || {
  printf 'Run the shared-disk copy of this script.\n' >&2
  exit 2
}

interval=${ZB_WAIT_INTERVAL:-15}
stable_required=${ZB_WAIT_STABLE_CHECKS:-4}
heartbeat_fresh=${ZB_READY_FRESH_SECONDS:-45}
wait_timeout=${ZB_WAIT_TIMEOUT_SECONDS:-21600}
[[ "$interval" =~ ^[1-9][0-9]*$ &&
   "$stable_required" =~ ^[1-9][0-9]*$ &&
   "$heartbeat_fresh" =~ ^[1-9][0-9]*$ &&
   "$wait_timeout" =~ ^[1-9][0-9]*$ ]] || exit 2

if [[ "$rank" == 0 ]]; then
  mkdir "$control"
else
  deadline=$((SECONDS + 300))
  while [[ ! -d "$control" ]]; do
    (( SECONDS < deadline )) || { printf 'A51 did not create %s\n' "$control" >&2; exit 1; }
    sleep 2
  done
fi

status() {
  printf '%s %s %s\n' "$(date -Is)" "$node" "$*" | tee "$control/$node.state"
}
failed() {
  compgen -G "$control/failed.*" >/dev/null
}
finish() {
  rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    : > "$control/failed.$node"
    status "stopped (exit $rc)"
  fi
  exit "$rc"
}
trap finish EXIT

wait_marker() {
  local marker=$1 limit=${2:-$wait_timeout} deadline=$((SECONDS + ${2:-$wait_timeout}))
  while [[ ! -e "$control/$marker" ]]; do
    failed && exit 1
    (( SECONDS < deadline )) || {
      status "timed out waiting for $marker"
      exit 1
    }
    sleep 10
  done
}

idle_now() {
  local stats ports
  stats=$(npu-smi info 2>/dev/null) || return 1
  ports=$(ss -H -lnt 'sport = :29555 or sport = :29556') || return 1
  [[ -z "$ports" ]] || return 1
  awk '
    /Ascend950DT/ { cards++ }
    /No running processes found in NPU [0-7][[:space:]]*\|/ { idle++ }
    END { exit !(cards == 8 && idle == 8) }
  ' <<< "$stats"
}

wait_until_idle() {
  local stage=$1 stable=0 deadline=$((SECONDS + wait_timeout))
  status "$stage: waiting for eight idle NPUs"
  while (( stable < stable_required )); do
    failed && exit 1
    (( SECONDS < deadline )) || {
      status "$stage: timed out waiting for idle NPUs"
      exit 1
    }
    if idle_now; then
      ((++stable))
      status "$stage: stable idle check $stable/$stable_required"
    else
      if (( stable > 0 )); then
        status "$stage: idle window interrupted; restarting checks"
      fi
      stable=0
    fi
    (( stable >= stable_required )) || sleep "$interval"
  done
}

wait_group_ready() {
  local stage=$1 nodes_csv=$2
  IFS=',' read -r -a expected_nodes <<< "$nodes_csv"
  status "$stage: synchronizing current idle heartbeats"
  while :; do
    failed && exit 1
    idle_now || {
      status "$stage: local NPU became busy while waiting for peers"
      wait_until_idle "$stage"
    }
    now=$(date +%s)
    printf '%s\n' "$now" > "$control/$stage-$node.ready.tmp"
    mv -f "$control/$stage-$node.ready.tmp" "$control/$stage-$node.ready"
    ready=true
    for peer in "${expected_nodes[@]}"; do
      marker=$control/$stage-$peer.ready
      if [[ ! -s "$marker" ]]; then
        ready=false
        break
      fi
      peer_time=$(< "$marker")
      if [[ ! "$peer_time" =~ ^[0-9]+$ ]] || (( now - peer_time > heartbeat_fresh )); then
        ready=false
        break
      fi
    done
    if [[ "$ready" == true ]]; then
      idle_now || continue
      return 0
    fi
    sleep 5
  done
}

configure_common() {
  unset ZB_PAIR_RUN ZB_MULTI_RUN ZB_SINGLE_RUN ZB_SCALE_NODES ZB_NODE_LIST ZB_PROFILE_FULL
  export ZB_JOB="$1"
  export ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
  export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10
  export ZB_FA2_UNPAD_CACHE=1 ZB_FA2_UNPAD_MARKERS=0 ZB_FA2_UNPAD_STATS=0
  export ZB_ZERO2_BUCKET_SIZE=auto ZB_ZERO2_OVERLAP_COMM=1 ZB_HCCL_DETERMINISTIC=1
  export ZB_MFU_PROFILE=fast ZB_MFU_SMOKE=0
  if [[ "$1" == profile ]]; then
    export ZB_PROFILE_FULL=1
  fi
}

run_single_job() {
  local job=$1
  configure_common "$job"
  export ZB_SINGLE_RUN="$id"
  status "single $job: starting 8 NPUs"
  bash "$here/benchmark-mfu.sh" |& tee "$control/single-$job-launcher.log"
}

run_multi_job() {
  local scale=$1 nodes_csv=$2 count=$3 job=$4
  configure_common "$job"
  export ZB_MULTI_RUN="$id-$scale" ZB_SCALE_NODES="$count" ZB_NODE_LIST="$nodes_csv"
  wait_group_ready "$scale-$job" "$nodes_csv"
  status "$scale $job: starting $((count * 8)) NPUs"
  bash "$here/benchmark-mfu.sh" |& tee "$control/$scale-$job-$node-launcher.log"
  : > "$control/$scale-$job-$node.done"
}

wait_group_done() {
  local scale=$1 nodes_csv=$2 job=$3
  IFS=',' read -r -a expected_nodes <<< "$nodes_csv"
  for peer in "${expected_nodes[@]}"; do
    wait_marker "$scale-$job-$peer.done"
  done
}

run_scale_stage() {
  local scale=$1 nodes_csv=$2 count=$3
  local nodes=(${nodes_csv//,/ })
  local member=false
  for candidate in "${nodes[@]}"; do
    [[ "$candidate" != "$node" ]] || member=true
  done
  [[ "$member" == true ]] || {
    wait_marker "$scale.done"
    return
  }

  for job in measure audit profile; do
    wait_until_idle "$scale-$job"
    run_multi_job "$scale" "$nodes_csv" "$count" "$job"
    wait_group_done "$scale" "$nodes_csv" "$job"
  done
  if [[ "$node" == "${nodes[0]}" ]]; then
    : > "$control/$scale.done"
  fi
}

generate_report() {
  local box=zhaobo-drivevla-a51
  local box_here=/work/zhaobo/${here#"$shared/"}
  local out=/work/zhaobo/artifacts/runs/mfu/scaling-$id
  status 'all profiles complete: generating unified report'
  docker exec "$box" bash --noprofile --norc -e -c '
    source /work/zhaobo/bootstrap/enter-env.sh
    export PYTHONPATH="$1/checkout:$1/checkout/reference/transformers/src:$PYTHONPATH"
    python "$1/report_scaling.py" \
      --measure single="$2/mfu-a51-$3-measure" \
      --measure dual="$2/mfu-a51-a52-$3-dual-measure" \
      --measure quad="$2/mfu-a51-a52-a53-a54-$3-quad-measure" \
      --audit single="$2/mfu-a51-$3-audit" \
      --audit dual="$2/mfu-a51-a52-$3-dual-audit" \
      --audit quad="$2/mfu-a51-a52-a53-a54-$3-quad-audit" \
      --profile single="$2/mfu-a51-$3-profile" \
      --profile dual="$2/mfu-a51-a52-$3-dual-profile" \
      --profile quad="$2/mfu-a51-a52-a53-a54-$3-quad-profile" \
      --ross-costs "$2/zhao-mfu01/ross.json" \
      --output "$4"
  ' _ "$box_here" "/work/zhaobo/artifacts/runs/mfu" "$id" "$out" \
    | tee "$control/comparison.stdout"
  : > "$control/completed"
}

status "registered in $control"
: > "$control/registered-$node"

if [[ "$node" == a51 ]]; then
  wait_until_idle single
  for job in measure audit profile; do
    run_single_job "$job"
    [[ "$job" != profile ]] || : > "$control/single.done"
  done
  run_scale_stage dual a51,a52 2
  if [[ ! -e "$control/dual.done" ]]; then
    : > "$control/dual.done"
  fi
elif [[ "$node" == a52 ]]; then
  wait_marker single.done
  run_scale_stage dual a51,a52 2
else
  wait_marker dual.done
fi

wait_marker dual.done
run_scale_stage quad a51,a52,a53,a54 4
if [[ "$node" == a51 ]]; then
  wait_marker quad.done
  generate_report
fi
status completed
