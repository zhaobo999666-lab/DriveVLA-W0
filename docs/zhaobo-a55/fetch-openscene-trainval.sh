#!/usr/bin/env bash
# 下载 OpenScene v1.1 trainval 相机分片（每片解压后删 tgz，可断点续跑）。
# 设计成在 A5 训练容器里跑（docker exec -d），因为下载与解压都在共享盘上完成：
#   docker exec -d zhaobo-drivevla-a52 bash /work/zhaobo/artifacts/code/a55-a56/fetch-openscene-trainval.sh 0 39 12
# 断点：每片成功后在 manifests/fetched-<tgz 名> 落 marker，重跑自动跳过。
set -Eeuo pipefail

S=${ZB_SHARED:-/work/zhaobo}            # 共享盘在容器内的挂载点
ROOT=${ZB_ROOT:-/work/zhaobo}
DL=$S/downloads/openscene
RGB=$S/data/raw-camera
MARK=$ROOT/manifests
LOG=$MARK/fetch-openscene-trainval.log
FIRST=${1:-0}
LAST=${2:-39}
JOBS=${3:-12}
REL=openscene-v1.1/openscene_sensor_trainval_camera
REPO=OpenDriveLab/OpenScene
REV=${ZB_DATAREV:-a76f840b65e972bc45e56c2adced897498e9a026}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=30
mkdir -p "$DL" "$RGB" "$MARK"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

fetch_one() {
  local i=$1 name tgz marker attempt
  name=openscene_sensor_trainval_camera_$i.tgz
  marker=$MARK/fetched-$name
  [[ -f $marker ]] && { log "skip  $name"; return 0; }
  tgz=$DL/$REL/$name
  for attempt in 1 2 3; do
    if hf download "$REPO" --repo-type dataset --revision "$REV" \
        --include "$REL/$name" --local-dir "$DL" --max-workers 1 --quiet >>"$LOG" 2>&1; then
      break
    fi
    log "retry $name (attempt $attempt)"
    sleep $((30 * attempt))
  done
  [[ -s $tgz ]] || { log "FAIL download $name"; return 1; }
  if ! tar --no-same-owner --no-same-permissions -xzf "$tgz" -C "$RGB" >>"$LOG" 2>&1; then
    log "FAIL extract $name"
    return 1
  fi
  rm -f "$tgz"          # 图像已落盘，tgz 不保留，省一半空间
  touch "$marker" "$MARK/extracted-$name"   # 第二个 marker 与 prepare-a55.sh 的约定对齐，避免它重复下载
  log "done  $name  raw-camera=$(du -sh "$RGB" | cut -f1)  free=$(df -h "$S" | awk 'NR==2{print $4}')"
}

export -f fetch_one log
export DL RGB MARK REL REPO REV LOG

log "start shards $FIRST..$LAST jobs=$JOBS"
seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} bash -c 'fetch_one {}'
log "shards $FIRST..$LAST finished (check FAIL lines above)"
