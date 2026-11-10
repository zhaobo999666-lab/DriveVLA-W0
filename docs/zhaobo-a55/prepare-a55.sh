#!/usr/bin/env bash
# Run on A55. No training, inference, or NPU tensor allocation.
set -Eeuo pipefail
umask 022
MODE=${1:?use environment, models or dataset}
case "$MODE" in environment|models|dataset) ;; *) exit 2;; esac
NODE=${ZB_NODE:-a55}
EXPECTED_HOST=${ZB_EXPECTED_HOST:-}
case "$NODE" in
  a55) EXPECTED_HOST=${EXPECTED_HOST:-liteserver-a5-2-00001.novalocal} ;;
  *)   # 其他节点（迁移后的 a52/a53 等）：peer 流程，必须显式给出预期主机名
       EXPECTED_HOST=${EXPECTED_HOST:?"peer node $NODE requires ZB_EXPECTED_HOST=<full hostname>"}
       # dataset/models 阶段是纯 CPU 的下载与 support.py 预处理（解包 + 生成 pkl），
       # 不分配 NPU 张量，所以允许在 peer 节点上跑；environment 会重建容器并校验
       # davinci 设备，仍只允许在 a55 上执行。
       [[ "$MODE" == environment || "$MODE" == dataset || "$MODE" == models ]] || exit 2 ;;
esac
HOST="/data1/zhaobo/drivevla-w0-$NODE"
SHARED=/mnt/sfs_turbo/zhaobo/drivevla-w0
WORK=/work/zhaobo
IMAGE=zhaobo/drivevla-w0:a55-cann9.2-torch2.11-py312-v1
BOX="zhaobo-drivevla-$NODE"
# 期望的代码提交以 release 的 REVISION 为唯一真相源（a55-a56.sh publish 写入）；
# 文件缺失时退回初始部署的提交。ZB_REV 可显式覆盖。
REVISION_FILE="$SHARED/artifacts/code/a55-a56/REVISION"
REV=${ZB_REV:-$(cat "$REVISION_FILE" 2>/dev/null || printf '%s' 2a87094e0ac52c425004eacdcced8734b3f68938)}
QREV=cc594898137f460bfe9f0759e9844b3ce807cfb5
SDREV=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
DATAREV=a76f840b65e972bc45e56c2adced897498e9a026
test "$(hostname)" = "$EXPECTED_HOST"
mountpoint -q /data1
mountpoint -q /mnt/sfs_turbo
test "$(realpath "$HOST")" = "$HOST"
test "$(realpath "$SHARED")" = "$SHARED"
exec 9>"$HOST/manifests/prepare-$MODE.lock"
flock -n 9
status() { printf '%s %s\n' "$(date -Is)" "$*" > "$HOST/manifests/prepare-$MODE.status"; }
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; if (( rc )); then status "FAILED exit=$rc"; else status READY; fi' EXIT
status RUNNING
stage() { status "RUNNING $*"; printf '\n[%s] %s\n' "$(date -Is)" "$*"; }
retry() {
  local i
  for i in 1 2 3; do
    if "$@"; then return 0; fi
    if (( i < 3 )); then sleep "$((60*i))"; fi
  done
  return 1
}
export PIP_DISABLE_PIP_VERSION_CHECK=1
PIP_MIRROR=${ZB_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}

if [[ $MODE == environment ]]; then
  if [[ "$NODE" != a55 ]]; then
    stage 'verify preloaded image'
    test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "${ZB_EXPECTED_IMAGE_ID:?load the verified A55 image first}"
  else
    stage 'build isolated ARM64 image'
    docker build --network host --platform linux/arm64 --build-arg "ZB_PIP_INDEX_URL=$PIP_MIRROR" -t "$IMAGE" "$HOST/bootstrap/image"
  fi
  docker image inspect "$IMAGE" > "$HOST/manifests/image.json"
  stage 'initialize pinned code from local bundle'
  BUNDLE="$SHARED/artifacts/code/DriveVLA-W0-$REV.bundle"
  REPO="$HOST/code/DriveVLA-W0"
  if [[ ! -e $REPO ]]; then
    git clone --single-branch --branch qwen3vl-backbone "$BUNDLE" "$REPO"
    git -C "$REPO" checkout --detach "$REV"
    git -C "$REPO" remote set-url origin https://github.com/mingchang93/DriveVLA-W0.git
  fi
  test "$(git -C "$REPO" rev-parse HEAD)" = "$REV"
  printf '%s\n' "$REV" > "$HOST/manifests/git-commit.txt"
  stage 'create own training container and verify imports without running training'
  devices=()
  for dev in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc; do
    test -c "$dev"
    devices+=(--device "$dev")
  done
  if [[ -c /dev/devmm_svm ]]; then devices+=(--device /dev/devmm_svm); fi
  mounts=(--mount "type=bind,src=$HOST,dst=$WORK")
  for dir in models downloads data artifacts; do
    mounts+=(--mount "type=bind,src=$SHARED/$dir,dst=$WORK/$dir")
  done
  if ! docker container inspect "$BOX" >/dev/null 2>&1; then
    docker run -d --name "$BOX" --label owner=zhaobo --network host --shm-size=16g \
      --ulimit memlock=-1:-1 --ulimit stack=67108864 "${devices[@]}" "${mounts[@]}" \
      --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
      --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
      "$IMAGE" sleep infinity
  fi
  test "$(docker inspect --format '{{index .Config.Labels "owner"}}' "$BOX")" = zhaobo
  test "$(docker inspect --format '{{.Image}}' "$BOX")" = "$(docker image inspect --format '{{.Id}}' "$IMAGE")"
  docker inspect --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' "$BOX" > "$HOST/manifests/mounts.txt"
  for dir in models downloads data artifacts; do
    grep -Fx "$SHARED/$dir -> $WORK/$dir" "$HOST/manifests/mounts.txt"
  done
  docker exec -i "$BOX" bash --noprofile --norc -e -s <<'ENV_CHECK'
source /work/zhaobo/bootstrap/enter-env.sh
env -u PYTHONPATH python -m pip install -e ./reference/transformers
env -u PYTHONPATH python -m pip check
python -m pip freeze > /work/zhaobo/manifests/pip-freeze.txt
DEVICE=cpu python -c 'import torch,torch_npu,transformers,deepspeed,diffusers; print(torch.__version__,torch_npu.__version__,transformers.__version__,transformers.__file__,deepspeed.__version__,diffusers.__version__)'
DEVICE=cpu python utils/train_qwen_vla.py --help > /work/zhaobo/manifests/train-help.txt
ENV_CHECK
  exit 0
fi

stage 'prepare isolated download and data tools'
VENV="$HOST/envs/download"
PY="$VENV/bin/python"
HF="$VENV/bin/hf"
(
  exec 8>"$HOST/manifests/download-venv.lock"
  flock -w 1800 8
  if [[ ! -f $HOST/manifests/download-venv.ready ]]; then
    test -x "$PY" || python3 -m venv "$VENV"
    unset PIP_EXTRA_INDEX_URL
    export PIP_CONFIG_FILE=/dev/null
    retry "$PY" -m pip install --index-url "$PIP_MIRROR" --timeout 45 --retries 2 \
      huggingface_hub==0.36.2 numpy==1.26.4 pillow==10.4.0 pyyaml==6.0.2 pyquaternion==0.9.9
    "$PY" -c 'import huggingface_hub,numpy,PIL,yaml,pyquaternion'
    test -x "$HF"
    touch "$HOST/manifests/download-venv.ready"
  fi
)
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export HF_HOME="$HOST/cache/huggingface-download" HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=120
if [[ $MODE == models ]]; then
Q="$SHARED/models/Qwen2.5-VL-7B-Instruct"
SD="$SHARED/models/stable-diffusion-v1-5"
stage 'download fixed Qwen2.5-VL weights'
retry "$HF" download Qwen/Qwen2.5-VL-7B-Instruct --revision "$QREV" --max-workers 1 --local-dir "$Q"
stage 'download fixed SD1.5 UNet, VAE and scheduler'
retry "$HF" download stable-diffusion-v1-5/stable-diffusion-v1-5 --revision "$SDREV" --max-workers 1 \
  --include unet/config.json unet/diffusion_pytorch_model.safetensors vae/config.json \
  vae/diffusion_pytorch_model.safetensors scheduler/scheduler_config.json model_index.json --local-dir "$SD"
stage 'check model shards and files'
"$PY" - "$Q" "$SD" <<'PY'
import json, struct, sys
from pathlib import Path
q, sd = map(Path, sys.argv[1:])
idx = json.loads((q / "model.safetensors.index.json").read_text())
for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
    json.loads((q / name).read_text())
for name in ("unet/config.json", "vae/config.json", "scheduler/scheduler_config.json", "model_index.json"):
    json.loads((sd / name).read_text())
for p in [*(q / name for name in set(idx["weight_map"].values())), sd / "unet/diffusion_pytorch_model.safetensors", sd / "vae/diffusion_pytorch_model.safetensors"]:
    with p.open("rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        assert 0 < n < 100_000_000, p
        header = json.loads(f.read(n))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    assert p.stat().st_size == 8 + n + end, p
    print(p.name, p.stat().st_size, "bytes OK")
PY
(
  cd "$SHARED/models"
  find Qwen2.5-VL-7B-Instruct stable-diffusion-v1-5 -type f -name '*.safetensors' \
    -exec sha256sum {} + > base-models.sha256.tmp
  mv base-models.sha256.tmp base-models.sha256
)
exit 0
fi
# 数据集划分：mini（默认，单 log 学习子集）或 trainval（全量）；用 ZB_SHARDS 控制下多少片
SPLIT=${ZB_SPLIT:-mini}
SHARDS=${ZB_SHARDS:-}
MAX_TRAIN=${ZB_MAX_TRAIN:-2048}
MAX_VAL=${ZB_MAX_VAL:-16}
PER_LOG_CAP=${ZB_PER_LOG_CAP:-0}
case "$SPLIT" in
  mini)     [[ -n $SHARDS ]] || SHARDS="0 1 2" ;;
  trainval) [[ -n $SHARDS ]] || SHARDS=$(seq -s ' ' 0 39) ;;
  *)        [[ -n $SHARDS ]] || { printf 'Set ZB_SHARDS for split %s\n' "$SPLIT" >&2; exit 2; } ;;
esac
DL="$SHARED/downloads/openscene"
DATA="$SHARED/data"
RGB="$DATA/raw-camera/openscene-v1.1/sensor_blobs/$SPLIT"
PREP="$DATA/prepared"
[[ "$SPLIT" == mini ]] || PREP="$DATA/prepared-$SPLIT"
# 预处理脚本以 release 副本为准（发布时一起同步），缺失时退回 bootstrap 里的旧副本
SUPPORT=${ZB_SUPPORT:-$SHARED/artifacts/code/a55-a56/support.py}
[[ -f $SUPPORT ]] || SUPPORT="$HOST/bootstrap/support.py"
if [[ ${ZB_REPREPARE:-0} != 1 && -s "$PREP/data_manifest.json" ]]; then
  for file in train.pkl val.pkl norm_stats.json train_tokens.yaml val_tokens.yaml; do test -s "$PREP/$file"; done
  test -d "$RGB"
  exit 0
fi
stage "prepare OpenScene $SPLIT metadatas and camera shards"
download_extract() {
  local rel=$1 dest=$2 marker="$HOST/manifests/extracted-${1##*/}"
  if [[ -f $marker ]]; then return 0; fi
  retry "$HF" download OpenDriveLab/OpenScene "$rel" --repo-type dataset --revision "$DATAREV" --local-dir "$DL"
  "$PY" - "$DL/$rel" <<'PY'
import sys, tarfile
from pathlib import PurePosixPath
with tarfile.open(sys.argv[1], "r|gz") as archive:
    for member in archive:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f"Unexpected archive member: {member.name!r}")
PY
  tar --no-same-owner --no-same-permissions -xzf "$DL/$rel" -C "$dest"
  touch "$marker"
}
download_extract "openscene-v1.1/openscene_metadata_$SPLIT.tgz" "$DATA/raw-logs"
for shard in $SHARDS; do
  stage "download and unpack OpenScene $SPLIT camera shard $shard"
  download_extract "openscene-v1.1/openscene_sensor_${SPLIT}_camera/openscene_sensor_${SPLIT}_camera_$shard.tgz" "$DATA/raw-camera" ||
    printf 'WARN: shard %s failed, continuing with the shards already unpacked\n' "$shard" >&2
done
if [[ ! -d $RGB ]]; then printf 'Camera directory missing: %s\n' "$RGB"; exit 1; fi
stage "generate train/val from $SPLIT (max_train=$MAX_TRAIN per_log_cap=$PER_LOG_CAP)"
# metadata 解压后统一落在 raw-logs/openscene-v1.1/meta_datas/<split>/；多个 split 共存时
# 从 raw-logs 根递归会把 mini 和 trainval 的同一 log 扫成两次（support.py 报 Duplicate
# raw log name），所以按 split 精确定位，找不到再退回 raw-logs 根。
LOGS="$DATA/raw-logs/openscene-v1.1/meta_datas/$SPLIT"
[[ -d $LOGS ]] || LOGS="$DATA/raw-logs"
tmp=$(mktemp -d "$DATA/.prepared.XXXXXX")
if "$PY" "$SUPPORT" prepare --logs "$LOGS" --rgb "$RGB" --out "$tmp" \
    --max-train "$MAX_TRAIN" --max-val "$MAX_VAL" --per-log-cap "$PER_LOG_CAP" \
    > "$HOST/manifests/prepare-data-$SPLIT.log" 2>&1; then
  cat "$HOST/manifests/prepare-data-$SPLIT.log"
  if [[ -e $PREP ]]; then mv -T "$PREP" "$PREP.bak-$(date +%Y%m%d-%H%M%S)"; fi
  mv -T "$tmp" "$PREP"
  printf 'Prepared dataset: %s\n' "$PREP"
  exit 0
fi
cat "$HOST/manifests/prepare-data-$SPLIT.log"
grep -Eq 'Need >=2 image-complete logs|Degenerate action quantiles|Empty train/validation split' "$HOST/manifests/prepare-data-$SPLIT.log" || exit 1
exit 1
