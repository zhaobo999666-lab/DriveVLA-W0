#!/usr/bin/env bash
# Run this same script on every node of the pair (any two A5 hosts).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="$HERE/a55-a56.sh"
MODE=${1:-train}

usage() {
  cat <<'USAGE'
Usage: bash start-training.sh [status|prepare|check|smoke|train] [run-id] [safe|fast] [batch] [qwen25|qwen3vl] [nodes=1|2]

  status   Print this node's health: env READY state, NPU busy devices, container, HCCL IP.
  prepare  On the source node (A55): export its image. On any peer node: load it and verify.
  check    Verify HCCL: nodes=2 -> 16 ranks on both nodes; nodes=1 -> 8 ranks on this node.
  smoke    Train 3 steps without saving.
  train    Train 50 steps.

Profiles (check/smoke/train):
  safe (default)  sdpa + ZeRO-3 CPU offload, no pin, workers 0 —— real50-04 verified baseline
  fast            fa2 + ZeRO-2 (no offload) + pinned loader, workers 2 —— npu-port.doc aligned;
                  train requires passing `check <id> fast ...` and `smoke <id> fast ...` on this node first
batch            per-NPU batch size (default 2)
model            qwen25 (default, Qwen2.5-VL-7B, max_pixels 150528) or
                 qwen3vl (Qwen3-VL-8B, max_pixels 903168 —— the npu-port MFU-best point,
                 directly comparable with utils/flops_mindspeed.py)
nodes            2 (default, dual-node 16 ranks, same command on both hosts) or
                 1 (single-node 8 ranks on this host only; npu-port best-point reference run)

Pairing any two A5 hosts (e.g. A52 + A53):
  Both hosts export the same ZB_PAIR=<id>, plus ZB_NODE=<short name> ZB_NODE_RANK=0/1.
  Rank 0 publishes its enp39s0f3 IP to the shared disk; rank 1 picks it up automatically,
  so ZB_MASTER_ADDR does not have to be copied by hand (it still overrides if you set it).
  Preflight without touching the NPUs: ZB_DRYRUN=1 bash start-training.sh check <id> fast 2 qwen3vl
USAGE
}

case "$MODE" in
  -h|--help) usage; exit 0 ;;
  status)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    exec bash "$LAUNCHER" status ;;
  prepare|check|smoke|train) ;;
  *) usage >&2; exit 2 ;;
esac
[[ -r "$LAUNCHER" ]] || { echo "ERROR: launcher not found: $LAUNCHER" >&2; exit 1; }

if [[ "$MODE" == prepare ]]; then
  [[ $# -eq 1 ]] || { usage >&2; exit 2; }
  # 老镜像的源节点（A55）负责导出；其余任意节点都走 peer 流程（load + 验收）
  case "$(hostname -s)" in
    liteserver-a5-2-00001) exec bash "$LAUNCHER" export-image ;;
    *) exec bash "$LAUNCHER" prepare-peer ;;
  esac
fi

case "$MODE" in
  check) default_id=comm01 ;;
  smoke) default_id=smoke01 ;;
  train) default_id=real50-01 ;;
esac
RUN_ID=${2:-$default_id}
PROFILE=${3:-safe}
BATCH=${4:-2}
MODEL=${5:-qwen25}
NODES=${6:-${ZB_NODES:-2}}
[[ "$PROFILE" == safe || "$PROFILE" == fast ]] ||
  { echo 'ERROR: profile must be safe or fast' >&2; exit 2; }
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] ||
  { echo 'ERROR: batch must be a positive integer' >&2; exit 2; }
[[ "$MODEL" == qwen25 || "$MODEL" == qwen3vl ]] ||
  { echo 'ERROR: model must be qwen25 or qwen3vl' >&2; exit 2; }
[[ "$NODES" == 1 || "$NODES" == 2 ]] ||
  { echo 'ERROR: nodes must be 1 or 2' >&2; exit 2; }
[[ $# -le 6 && "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] ||
  { echo 'ERROR: invalid run ID; use letters, digits, underscores or hyphens' >&2; exit 2; }

if [[ "$MODE" == train && "${ZB_DRYRUN:-0}" != 1 ]]; then
  run_dir="/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/a55-a56/train-$RUN_ID/navsim"
  if [[ -e "$run_dir/trainer_state.json" ]] || compgen -G "$run_dir/checkpoint-*" >/dev/null; then
    echo "ERROR: run $RUN_ID already has training state; use a new run ID." >&2
    echo "To resume deliberately, run bash a55-a56.sh train $RUN_ID on both hosts." >&2
    exit 1
  fi
fi

exec bash "$LAUNCHER" "$MODE" "$RUN_ID" "$PROFILE" "$BATCH" "$MODEL" "$NODES"
