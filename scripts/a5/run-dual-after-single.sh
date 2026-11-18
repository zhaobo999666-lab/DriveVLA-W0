#!/usr/bin/env bash
# After a successful A53 baseline, wait until both hosts are stably idle and run dual A/B.
set -euo pipefail

case "$(hostname -s)" in
  liteserver-a5-00003) node=a53; peer=a54 ;;
  liteserver-a5-00004) node=a54; peer=a53 ;;
  *) printf 'Run on A53 or A54.\n' >&2; exit 2 ;;
esac
[[ $# == 2 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,48}$ &&
   "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,38}$ ]] || {
  printf 'Usage: %s SINGLE_RUN_ID DUAL_RUN_ID\n' "$0" >&2; exit 2;
}

single_id=$1
dual_id=$2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
single_control=$root/wait-single-$single_id
control=$root/dual-after-single-$dual_id
if [[ "$node" == a53 ]]; then
  mkdir "$control"
else
  for _ in $(seq 1 120); do [[ -d "$control" ]] && break; sleep 2; done
  [[ -d "$control" ]] || exit 1
fi

status() { printf '%s %s %s\n' "$(date -Is)" "$node" "$*" | tee "$control/$node.state"; }
finish() {
  rc=$?
  if [[ $rc != 0 ]]; then
    printf '%s\n' "$rc" > "$control/$node.failed"
    status "stopped (exit $rc)"
  fi
}
trap finish EXIT

idle_now() {
  local stats ports
  stats=$(npu-smi info) || return 1
  ports=$(ss -H -lnt 'sport = :29555 or sport = :29556') || return 1
  [[ -z "$ports" ]] || return 1
  awk '
    /Ascend950DT/ { cards++ }
    /No running processes found in NPU [0-7][[:space:]]*\|/ { idle++ }
    END { exit !(cards == 8 && idle == 8) }
  ' <<< "$stats"
}

status "waiting for successful A53 single run: $single_id"
while [[ ! -s "$single_control/exit" ]]; do sleep 20; done
[[ $(< "$single_control/exit") == 0 ]] || {
  status 'single run failed; dual run cancelled'; exit 1;
}

own_ready=$control/$node.ready
peer_ready=$control/$peer.ready
stable=0
status 'single run succeeded; waiting for both hosts to become stably idle'
while :; do
  if idle_now; then
    ((++stable))
  else
    stable=0
  fi
  if (( stable >= 4 )); then
    while idle_now; do
      now=$(date +%s)
      printf '%s\n' "$now" > "$own_ready.tmp"
      mv "$own_ready.tmp" "$own_ready"
      if [[ -s "$peer_ready" ]]; then
        peer_time=$(< "$peer_ready")
        if [[ "$peer_time" =~ ^[0-9]+$ && $((now - peer_time)) -le 15 ]]; then
          status 'both hosts have a current idle heartbeat; starting dual A/B'
          break 2
        fi
      fi
      sleep 5
    done
    stable=0
    status 'local idle window interrupted; waiting again'
  fi
  sleep 15
done

idle_now || { status 'local host became busy before launch; aborting safely'; exit 1; }
bash "$here/run-zero2-ab.sh" "$dual_id"
status completed
