#!/usr/bin/env bash
# Run on the A5-series host, not inside the container.
# 节点身份三种方式（优先级从高到低）：
#   1) 单机 ZB_NODES=1 —— 任意 A5 节点独立跑 8 卡，rank 固定 0
#   2) 环境变量 ZB_NODE=<短名> ZB_NODE_RANK=<0|1>
#   3) 主机名自动识别（历史 A55/A56 映射）
# 双机配对推荐再给同一个 ZB_PAIR=<配对ID>：rank0 把本机 enp39s0f3 IP 写进共享盘，
# rank1 自动读取，不必手抄对方 IP。也可继续用 ZB_MASTER_ADDR 显式指定。
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
# 未显式给 ZB_NODE 时，按 /data1 上的工作区目录反推短名（a52/a53/a54…），否则退回主机短名
infer_node() {
  local cand
  cand=$(ls -d /data1/zhaobo/drivevla-w0-* 2>/dev/null | head -1)
  if [[ -n "$cand" ]]; then printf '%s' "${cand##*/drivevla-w0-}"; else hostname -s; fi
}
MODE=${1:-}
case "$MODE" in
  ''|export-image|prepare-peer|check|smoke|train|status) ;;
  *) die "Unknown mode '$MODE' (export-image|prepare-peer|check|smoke|train|status)" ;;
esac
NODE=""; NODE_RANK=""
case "$(hostname -s)" in
  liteserver-a5-2-00001) NODE=a55; NODE_RANK=0 ;;
  liteserver-a5-2-00002) NODE=a56; NODE_RANK=1 ;;
esac
: "${NODE:=${ZB_NODE:-}}"; : "${NODE_RANK:=${ZB_NODE_RANK:-}}"
NODES=${6:-${ZB_NODES:-2}}
[[ "$NODES" == 1 || "$NODES" == 2 ]] || die "Nodes must be 1 or 2 (got '$NODES')"
# 节点名缺省按 /data1 工作区目录反推（a52/a53/a54…），所以登录后只需给 rank
: "${NODE:=$(infer_node)}"
# 单机模式：任何 A5 节点都可作为独立主机运行，rank 固定 0
if [[ "$NODES" == 1 ]]; then
  NODE_RANK=0
fi
# status 只是本机体检，未配 rank 也能跑
if [[ "$MODE" == status ]]; then
  NODE_RANK=${NODE_RANK:-0}
fi
[[ -n "$NODE_RANK" ]] ||
  die "Unknown rank; set ZB_NODE_RANK=<0|1> (single-node: ZB_NODES=1)"
HCCL_IP=$(ip -o -4 addr show dev enp39s0f3 2>/dev/null | awk '{sub(/\/.+$/,"",$4); print $4; exit}')
[[ -n "$HCCL_IP" || "$MODE" == status ]] || die "No IPv4 address on enp39s0f3"
MASTER_ADDR=${ZB_MASTER_ADDR:-}
if [[ "$NODE_RANK" == 0 ]]; then
  MASTER_ADDR=$HCCL_IP
fi
# rank1 的会合地址只在 check/smoke/train 需要（prepare-peer 不会合），见下方分支校验
HERE=$(cd "$(dirname "$0")" && pwd)
SHARED=/mnt/sfs_turbo/zhaobo/drivevla-w0
HOST="/data1/zhaobo/drivevla-w0-$NODE"
BOX="zhaobo-drivevla-$NODE"
IMAGE=zhaobo/drivevla-w0:a55-cann9.2-torch2.11-py312-v1
RELEASE="$SHARED/artifacts/code/a55-a56"
EXPORT="$SHARED/artifacts/images/a55-qwen-v1"
REV=2a87094e0ac52c425004eacdcced8734b3f68938
RUNS="$SHARED/artifacts/runs/a55-a56"
PAIR=${ZB_PAIR:-}
PAIR_DIR=""
if [[ -n "$PAIR" ]]; then
  [[ "$PAIR" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]] ||
    die "ZB_PAIR must be 1-32 chars of letters, digits, underscore or hyphen (got '$PAIR')"
  PAIR_DIR="$SHARED/artifacts/runs/pairs/$PAIR"
fi
mountpoint -q /mnt/sfs_turbo || die "Shared filesystem is not mounted"
mountpoint -q /data1 || die "/data1 is not mounted"
[[ "$HERE" == "$RELEASE" ]] || die "Run the launcher from $RELEASE"

case "$MODE" in
  status)
    # 未显式给 ZB_NODE 时，按 /data1 上的工作区目录反推短名（a52/a53/a54…）
    if [[ -z "${ZB_NODE:-}" ]]; then
      cand=$(ls -d /data1/zhaobo/drivevla-w0-* 2>/dev/null | head -1)
      [[ -n "$cand" ]] && { NODE=${cand##*/drivevla-w0-}; HOST="/data1/zhaobo/drivevla-w0-$NODE"; BOX="zhaobo-drivevla-$NODE"; }
    fi
    busy=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ / [0-9]+' |
      awk -F' / ' '$2+0 >= 16384 && $1+0 > 2048 {c++} END{print c+0}')
    env_state=$(tail -1 "$HOST/manifests/prepare-environment.status" 2>/dev/null || echo MISSING)
    box=$(docker ps --filter "name=^${BOX}$" --format '{{.Status}}' 2>/dev/null || true)
    printf 'node=%-5s host=%-20s hccl=%-14s env=%-6s npu_busy=%s/8 container=%s\n' \
      "$NODE" "$(hostname -s)" "${HCCL_IP:--}" "${env_state##* }" "$busy" "${box:-none}"
    ;;
  export-image)
    [[ "$NODE" == a55 ]] || die "Export the image on A55"
    grep -q ' READY$' "$HOST/manifests/prepare-environment.status" ||
      die "A55 environment has not passed validation"
    if [[ -e "$EXPORT" ]]; then
      [[ -s "$EXPORT/image.tar" && -s "$EXPORT/SHA256SUMS" && -s "$EXPORT/image.id" && -s "$EXPORT/enter-env.sh" ]] ||
        die "Existing export is incomplete: $EXPORT"
      printf 'Existing export: %s\n' "$EXPORT"
      exit 0
    fi
    image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
    [[ $(docker inspect --format '{{.Image}}' "$BOX") == "$image_id" ]] ||
      die "A55 container does not match the image tag"
    stage=$(mktemp -d "$SHARED/artifacts/images/.a55-qwen-v1.XXXXXX")
    printf 'Exporting image to %s\n' "$stage"
    docker save -o "$stage/image.tar" "$IMAGE"
    printf '%s\n' "$image_id" > "$stage/image.id"
    install -m 0644 "$HOST/bootstrap/enter-env.sh" "$stage/enter-env.sh"
    (cd "$stage"; sha256sum image.tar image.id enter-env.sh > SHA256SUMS)
    mv -T -n "$stage" "$EXPORT"
    [[ ! -e "$stage" ]] || die "Another export already exists; staged copy: $stage"
    printf 'Image export ready: %s\n' "$EXPORT"
    ;;
  prepare-peer)
    [[ "$NODE" != a55 ]] || die "prepare-peer runs on the peer node (not A55)"
    [[ -s "$EXPORT/SHA256SUMS" ]] || die "First run export-image on A55"
    (cd "$EXPORT"; sha256sum -c SHA256SUMS)
    expected=$(cat "$EXPORT/image.id")
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      docker load -i "$EXPORT/image.tar"
    fi
    [[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$expected" ]] ||
      die "A56 already has a different image under the same tag"
    mkdir -p "$HOST"/{bootstrap,code,manifests,runs}
    install -m 0644 "$EXPORT/enter-env.sh" "$HOST/bootstrap/enter-env.sh"
    ZB_NODE="$NODE" ZB_EXPECTED_HOST="$(hostname)" ZB_EXPECTED_IMAGE_ID="$expected" bash "$HERE/prepare-a55.sh" environment
    ;;
  check|smoke|train)
    RUN_ID=${2:-}
    PROFILE=${3:-safe}
    BATCH=${4:-2}
    [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] ||
      die "Supply the same run ID on both nodes (letters, digits, underscores or hyphens)"
    [[ "$PROFILE" == safe || "$PROFILE" == fast ]] || die "Profile must be safe or fast (got '$PROFILE')"
    [[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || die "Batch size must be a positive integer (got '$BATCH')"
    MODEL=${5:-qwen25}
    [[ "$MODEL" == qwen25 || "$MODEL" == qwen3vl ]] || die "Model must be qwen25 or qwen3vl (got '$MODEL')"
    NODES=${6:-${ZB_NODES:-2}}
    [[ "$NODES" == 1 || "$NODES" == 2 ]] || die "Nodes must be 1 or 2 (got '$NODES')"
    if [[ "$NODES" == 1 ]]; then
      NODE_RANK=0; MASTER_ADDR=$HCCL_IP
    elif [[ "$NODE_RANK" == 0 ]]; then
      MASTER_ADDR=$HCCL_IP
      # 配对握手：把本机会合 IP 写到共享盘，供 rank1 自动读取（免手抄 IP）
      if [[ -n "$PAIR_DIR" ]]; then
        mkdir -p "$PAIR_DIR"
        printf '%s\n' "$HCCL_IP" > "$PAIR_DIR/master-$RUN_ID.ip.tmp"
        mv -f "$PAIR_DIR/master-$RUN_ID.ip.tmp" "$PAIR_DIR/master-$RUN_ID.ip"
        printf '[pair] %s rank0 advertises %s for run %s\n' "$NODE" "$HCCL_IP" "$RUN_ID"
      fi
    elif [[ -z "$MASTER_ADDR" ]]; then
      # rank1：优先 ZB_MASTER_ADDR，其次等 rank0 写出的握手文件
      [[ -n "$PAIR_DIR" ]] ||
        die "Rank-1 host must set ZB_MASTER_ADDR=<rank0 enp39s0f3 IP>, or ZB_PAIR=<same pair id> on both nodes for auto discovery"
      mkdir -p "$PAIR_DIR"
      for _ in $(seq 1 60); do
        [[ -s "$PAIR_DIR/master-$RUN_ID.ip" ]] && MASTER_ADDR=$(head -1 "$PAIR_DIR/master-$RUN_ID.ip") && break
        sleep 5
      done
      [[ -n "$MASTER_ADDR" ]] ||
        die "Timed out waiting for $PAIR_DIR/master-$RUN_ID.ip; start rank0 with the same ZB_PAIR and run id first"
      printf '[pair] %s rank1 discovered master %s for run %s\n' "$NODE" "$MASTER_ADDR" "$RUN_ID"
    fi
    if [[ "$MODEL" == qwen3vl ]]; then
      [[ -d "$SHARED/models/Qwen3-VL-8B-Instruct" ]] ||
        die "Qwen3-VL-8B-Instruct weights missing; download to $SHARED/models/Qwen3-VL-8B-Instruct (container /work/zhaobo/models/Qwen3-VL-8B-Instruct)"
    fi
    # fast 档必须先跑通同 profile 的 check 与 smoke（宿主机日志留有标记）；
    # 单机（NODES=1）只看本机日志，双机需两台都跑过。预演（ZB_DRYRUN=1）跳过此门禁。
    if [[ "$PROFILE" == fast && "$MODE" == train && "${ZB_DRYRUN:-0}" != 1 ]]; then
      grep -qs 'HCCL OK' "$HOST/runs/a55-a56"/check-*"$PROFILE"*.log ||
        die "fast profile: run 'bash a55-a56.sh check <id> $PROFILE $BATCH $MODEL $NODES' on this node first"
      grep -qs 'SMOKE OK' "$HOST/runs/a55-a56"/smoke-*"$PROFILE"*.log ||
        die "fast profile: run 'bash a55-a56.sh smoke <id> $PROFILE $BATCH $MODEL $NODES' on this node first"
    fi
    grep -q ' READY$' "$HOST/manifests/prepare-environment.status" || die "Environment not ready on $NODE"
    [[ $(git -C "$HOST/code/DriveVLA-W0" rev-parse HEAD) == "$REV" ]] || die "Unexpected project revision"
    [[ $(docker inspect --format '{{.HostConfig.NetworkMode}}' "$BOX") == host ]] ||
      die "Container must use host networking"
    ip -o -4 addr show dev enp39s0f3 | grep -Fq " $HCCL_IP/" || die "enp39s0f3 address changed during run"
    # 共享盘多人排队：同一个 run ID 已有产物时拒绝覆盖，避免撞掉别人的结果
    OUT_HOST="$RUNS/$MODE-$RUN_ID"
    if [[ -d "$OUT_HOST" && -n "$(ls -A "$OUT_HOST" 2>/dev/null)" ]]; then
      die "Run ID '$RUN_ID' already has output at $OUT_HOST; choose a new run ID"
    fi
    port=29555
    [[ "$MODE" != check ]] || port=29556
    DRYRUN=${ZB_DRYRUN:-0}
    if [[ "$DRYRUN" == 1 ]]; then
      printf '[dry-run] node=%s rank=%s hccl=%s master=%s port=%s nodes=%s\n' \
        "$NODE" "$NODE_RANK" "$HCCL_IP" "$MASTER_ADDR" "$port" "$NODES"
      printf '[dry-run] mode=%s run=%s profile=%s model=%s batch=%s pair=%s\n' \
        "$MODE" "$RUN_ID" "$PROFILE" "$MODEL" "$BATCH" "${PAIR:-none}"
      printf '[dry-run] output=%s\n' "$OUT_HOST"
      exit 0
    fi
    [[ -z $(ss -H -lnt "sport = :$port") ]] || die "Port $port is busy on $NODE"
    # 空闲判定：兼容两代 npu-smi 输出
    #   旧版（910B/24.x 之前）逐卡打印 "No running processes found in NPU"
    #   新版（25.1.rc2 / 950DT）只给 HBM 用量行 "59257 / 98304"，需按占用判断
    idle_markers=$(npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU' || true)
    busy=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ / [0-9]+' |
      awk -F' / ' '$2+0 >= 16384 && $1+0 > 2048 {c++} END{print c+0}')
    if [[ "$idle_markers" != 8 && "$busy" != 0 ]]; then
      die "NPU busy on $NODE: $busy/8 devices hold >2 GB HBM (another job, e.g. vLLM, is resident); wait or coordinate"
    fi
    mkdir -p "$HOST/runs/a55-a56"
    docker exec -i "$BOX" bash --noprofile --norc -e -s -- \
      "$MODE" "$RUN_ID" "$NODE_RANK" "$MASTER_ADDR" "$port" "$HCCL_IP" "$PROFILE" "$BATCH" \
      "$NODE" "$MODE" "$MODEL" "$NODES" <<'INSIDE' \
      2>&1 | tee -a "$HOST/runs/a55-a56/$MODE-$RUN_ID-$PROFILE-$MODEL-b$BATCH-n$NODES.log"
mode=$1 run_id=$2 rank=$3 master=$4 port=$5 hccl_ip=$6 profile=$7 batch=$8 node=$9 logmode=${10} model=${11} nodes=${12}
source /work/zhaobo/bootstrap/enter-env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=enp39s0f3 HCCL_IF_IP="$hccl_ip" HCCL_CONNECT_TIMEOUT=120
release=/work/zhaobo/artifacts/code/a55-a56
if [[ "$mode" == check ]]; then
  mkdir -p /work/zhaobo/runs/a55-a56
  torchrun --nnodes="$nodes" --nproc_per_node=8 --node_rank="$rank" \
    --master_addr="$master" --master_port="$port" --rdzv_conf=timeout=120 "$release/check-hccl.py" \
    && { echo " HCCL OK " | tee -a "/work/zhaobo/runs/a55-a56/check-latest-$profile.log"; exit 0; }
  exit 1
fi
mkdir -p /work/zhaobo/artifacts/runs/a55-a56 /work/zhaobo/runs/a55-a56
install -m 0644 "$release/train_qwen_vla_navsim.sh" ./scripts/scripts_train/train_qwen_vla_navsim.sh
steps=50; warmup=5; save=true
if [[ "$mode" == smoke ]]; then steps=3; warmup=0; save=false; fi
# 档位（对照 npu-port.zh-CN.md 附录）：
#   safe = sdpa + ZeRO-3 CPU offload + 无 pin（real50-04 验证过的稳定基线）
#   fast = fa2 + ZeRO-2（无 offload）+ pinned loader —— npu-port 实测 fa2 +83%、ZeRO-2 优于 ZeRO-3
#   两者共用脚本默认的 CPU_AFFINITY_CONF/expandable_segments/融合 RMSNorm+RoPE/VALID_CE
# 模型（第 5 个参数）：
#   qwen25  = Qwen2.5-VL-7B-Instruct，max_pixels 150528（默认）
#   qwen3vl = Qwen3-VL-8B-Instruct，max_pixels 903168（npu-port 最佳点 L≈1774，
#             ViT 每图 3432 patch tokens；与 utils/flops_mindspeed.py 口径一致）
if [[ "$profile" == fast ]]; then
  ATTN=fa2; DS_CFG="$PWD/scripts/sft/zero2.json"; PIN=true; WORKERS=2
else
  ATTN=sdpa; DS_CFG="$PWD/scripts/sft/zero3_offload.json"; PIN=false; WORKERS=0
fi
if [[ "$model" == qwen3vl ]]; then
  MODEL_PATH=/work/zhaobo/models/Qwen3-VL-8B-Instruct; MAXPX=903168
else
  MODEL_PATH=/work/zhaobo/models/Qwen2.5-VL-7B-Instruct; MAXPX=150528
fi
echo "[a55-a56] node=$node mode=$logmode profile=$profile model=$model batch=$batch nodes=$nodes attn=$ATTN ds=$(basename "$DS_CFG") pin=$PIN workers=$WORKERS max_pixels=$MAXPX"
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
  --nnodes "$nodes" --node_rank "$rank" --master_addr "$master" --master_port "$port" --ngpus 8 \
  --model_name_or_path "$MODEL_PATH" \
  --sd_model_path /work/zhaobo/models/stable-diffusion-v1-5 \
  --data_path /work/zhaobo/data/prepared/train.pkl --data_root "$RGB_ROOT" \
  --device npu --attn_type "$ATTN" --fp bf16 --batch_size "$batch" --grad_accum 1 \
  --max_pixels "$MAXPX" \
  --num_workers "$WORKERS" --pin_memory "$PIN" --deepspeed_config "$DS_CFG" \
  --max_steps "$steps" --warmup_steps "$warmup" --logging_steps 1 --save_model "$save" \
  --save_steps 25 --save_total_limit 1 --report_to tensorboard \
  --output_dir "/work/zhaobo/artifacts/runs/a55-a56/$mode-$run_id" --exp_name navsim
rc=$?
if [[ "$mode" == smoke && $rc -eq 0 ]]; then
  echo " SMOKE OK " | tee -a "/work/zhaobo/runs/a55-a56/smoke-latest-$profile.log"
fi
exit $rc
INSIDE
    ;;
  *) die "Usage: bash $0 status | export-image | prepare-peer | {check|smoke|train} <run-id> [safe|fast] [batch] [qwen25|qwen3vl] [nodes=1|2]
       双机配对（任意两台 A5，例如 A52+A53）：
         两台都 export ZB_PAIR=<配对ID>，并分别 export ZB_NODE=<短名> ZB_NODE_RANK=0/1；
         rank0 会把本机 IP 写到共享盘，rank1 自动读取，无需手抄 ZB_MASTER_ADDR。
       单机：nodes=1（或 ZB_NODES=1），忽略 ZB_NODE_RANK/ZB_MASTER_ADDR。
       预演：ZB_DRYRUN=1 只打印解析出的配置，不校验 NPU/端口、不启动训练。" ;;
esac
