#!/usr/bin/env bash
# Qwen3-VL + ROSS performance baseline on one reserved 8-NPU A5 host.
set -euo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

case $(hostname -s) in
  liteserver-a5-00002) node=a52 ;;
  liteserver-a5-00003) node=a53 ;;
  liteserver-a5-00004) node=a54 ;;
  *) die 'Run this on a prepared A52, A53 or A54 host (outside Docker).' ;;
esac
[[ $# -le 1 && ( $# -eq 0 || $1 == --dry-run ) ]] || die 'Usage: bash benchmark-mfu.sh [--dry-run]'
shared=/mnt/sfs_turbo/zhaobo/drivevla-w0
host=/data1/zhaobo/drivevla-w0-$node
box=zhaobo-drivevla-$node
release=$shared/artifacts/code/a55-a56
repo=$host/code/DriveVLA-W0
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ "$here" == "$shared/"* ]] || die 'Run the copy on the shared drive.'
helper=/work/zhaobo/${here#"$shared/"}/benchmark_mfu.py
nodes=1 rank=0 job=${ZB_JOB:-measure}
[[ "$job" == measure || "$job" == profile || "$job" == smoke ]] || die 'Invalid ZB_JOB.'
if [[ -n ${ZB_PAIR_RUN:-} ]]; then
  [[ "$ZB_PAIR_RUN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || die 'Invalid ZB_PAIR_RUN.'
  nodes=2
  case $node in a53) rank=0;; a54) rank=1;; *) die 'Dual run requires A53/A54.';; esac
  export ZB_MFU_PROFILE=fast ZB_MFU_SMOKE=0
fi
model=/work/zhaobo/models/Qwen3-VL-8B-Instruct
sd=/work/zhaobo/models/stable-diffusion-v1-5
data=/work/zhaobo/data/prepared-trainval/train.pkl
rgb=/work/zhaobo/data/raw-camera/openscene-v1.1/sensor_blobs/trainval
batch=${ZB_MBS:-2}
[[ "$batch" =~ ^[1-9][0-9]{0,2}$ && "$batch" -ge 2 ]] || die 'ZB_MBS must be an integer >= 2.'
case ${ZB_MFU_PROFILE:-both} in
  both) profiles=(safe fast) ;;
  safe|fast) profiles=("$ZB_MFU_PROFILE") ;;
  *) die 'ZB_MFU_PROFILE must be both, safe or fast.' ;;
esac
steps=${ZB_MFU_STEPS:-50}
warmup=${ZB_MFU_WARMUP:-5}
gateup=${ZB_FUSED_GATEUP:-0}
smoke=${ZB_MFU_SMOKE:-1}
fast_zero=${ZB_FAST_ZERO_STAGE:-2}
gc_skip=${ZB_GC_SKIP_TEXT_LAST:-0}
[[ "$gc_skip" =~ ^(0|[1-9][0-9]?)$ ]] || die 'Invalid ZB_GC_SKIP_TEXT_LAST.'
[[ "$steps" =~ ^[1-9][0-9]{0,2}$ && "$warmup" =~ ^(0|[1-9][0-9]{0,2})$ && "$steps" -gt "$warmup" ]] ||
  die 'Set 0 <= ZB_MFU_WARMUP < ZB_MFU_STEPS <= 999.'
[[ "$gateup" == 0 || "$gateup" == 1 ]] || die 'ZB_FUSED_GATEUP must be 0 or 1.'
[[ "$smoke" == 0 || "$smoke" == 1 ]] || die 'ZB_MFU_SMOKE must be 0 or 1.'
[[ "$fast_zero" == 2 || "$fast_zero" == 3 ]] || die 'ZB_FAST_ZERO_STAGE must be 2 or 3.'
if [[ " ${profiles[*]} " == *' fast '* && "$fast_zero" == 2 ]]; then
  [[ -s "$here/npu_zero2_compat.py" && -x "$here/mfu-fast-bin/torchrun" ]] || die 'ZeRO-2 compatibility entry missing.'
fi
revision=$(git -C "$repo" rev-parse HEAD)
[[ "$revision" == "$(< "$release/REVISION")" ]] || die 'Local repo and shared release are out of sync.'
cmp -s "$repo/scripts/scripts_train/train_qwen_vla_navsim.sh" "$release/train_qwen_vla_navsim.sh" ||
  die 'Local training script differs from the shared release.'
grep -q ' READY$' "$host/manifests/prepare-environment.status" || die 'Training environment not ready.'
[[ $(docker inspect -f '{{.State.Running}} {{.HostConfig.NetworkMode}}' "$box") == 'true host' ]] ||
  die 'Training container must be running with host networking.'
ip=$(ip -o -4 addr show dev enp39s0f3 | awk '{sub(/\/.+$/, "", $4); print $4; exit}')
[[ -n "$ip" ]] || die 'No IPv4 on enp39s0f3.'
master=$ip
if [[ "$nodes" == 2 ]]; then
  master=10.0.16.231
  [[ "$rank" != 0 || "$ip" == "$master" ]] || die 'A53 rendezvous IP changed.'
  [[ "$rank" != 1 || "$ip" == 10.0.16.118 ]] || die 'A54 network IP changed.'
fi
if [[ "$job" == profile ]]; then
  [[ "$steps" -ge 30 ]] || die 'Profiler requires at least 30 steps.'
fi
printf 'nodes=%s rank=%s master=%s:29555 phase=%s\n' "$nodes" "$rank" "$master" "$job"
printf 'gc_skip_text_last=%s\n' "$gc_skip"
docker exec "$box" bash -e -c '
  test -r "$1" && test -s "$2/config.json" && test -d "$3/unet" &&
  test -d "$3/vae" && test -s "$4" && test -d "$5"
' _ "$helper" "$model" "$sd" "$data" "$rgb" || die 'Benchmark helper, model or dataset missing inside the container.'
docker exec "$box" bash --noprofile --norc -e -c '
  source /work/zhaobo/bootstrap/enter-env.sh
  python -c "import importlib.util as u; missing = [p for p in (\"torch\", \"torch_npu\", \"transformers\", \"tokenizers\", \"deepspeed\", \"diffusers\") if not u.find_spec(p)]; raise SystemExit(\"Missing packages: \" + \", \".join(missing) if missing else 0)"
' || die 'Recreated container may need the project environment preparation again.'
status=$(cd "$release" && ZB_NODE="$node" bash start-training.sh status)
printf '%s\n' "$status"
[[ "$status" == *'sync=ok'* ]] || die 'Shared release files are out of sync.'
printf 'Plan: %s, Qwen3-VL-8B + ROSS, trainval subset, MBS=%s GBS=%s, profiles=%s, steps=%s warmup=%s, fast_gateup=%s fast_zero=%s, no checkpoints\n' \
  "$node" "$batch" "$((batch * 8 * nodes))" "${profiles[*]}" "$steps" "$warmup" "$gateup" "$fast_zero"
[[ ${1:-} != --dry-run ]] || exit 0

exec 9>"$host/.mfu-benchmark.lock"
flock -n 9 || die 'Another MFU benchmark is already running on this node.'

assert_idle() {
  local stats ports
  stats=$(npu-smi info) || die 'Cannot read NPU status.'
  # A54 can report about 4.8 GiB of baseline HBM on idle cards. Prefer the
  # explicit process table when it covers all eight cards; keep a conservative
  # 2 GiB limit on versions that do not report explicit idle markers.
  if ! awk '
    /Ascend950DT/ {cards++}
    /No running processes found in NPU [0-7][[:space:]]*\|/ {idle[$(NF-1)]=1}
    /\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|/ {processes++}
    {
      s=$0
      while (match(s, /[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+/)) {
        split(substr(s,RSTART,RLENGTH), p, "/")
        if (p[2]+0 >= 16384) {
          rows++; if (p[1]+0 > 2048) busy++; if (p[1]+0 > 8192) high++
        }
        s=substr(s,RSTART+RLENGTH)
      }
    }
    END {
      for (i in idle) idle_count++
      exit !(cards==8 && rows==8 && !processes && (!busy || (idle_count==8 && !high)))
    }
  ' <<< "$stats"; then
    die 'Eight idle Ascend950DT devices could not be verified. Check npu-smi info and retry when reserved and idle.'
  fi
  ports=$(ss -H -lnt 'sport = :29555 or sport = :29556') || die 'Cannot inspect training ports.'
  [[ -z "$ports" ]] || die 'Training or HCCL port already in use.'
}

run_id=mfu-$node-$(date +%Y%m%d-%H%M%S)-$$
if [[ "$nodes" == 2 ]]; then run_id=mfu-a53-a54-$ZB_PAIR_RUN-$job; fi
out=artifacts/runs/mfu/$run_id
out_host=$shared/$out
out_box=/work/zhaobo/$out
assert_idle
mkdir -p "$shared/artifacts/runs/mfu"
if [[ "$rank" == 0 ]]; then
mkdir "$out_host"
printf 'node=%s\nrevision=%s\nmbs=%s\ngbs=%s\nsteps=%s\nwarmup=%s\nfast_gateup=%s\nfast_zero=%s\npeak_flops_assumed=486000000000000\n' \
  "$node" "$revision" "$batch" "$((batch * 8 * nodes))" "$steps" "$warmup" "$gateup" "$fast_zero" > "$out_host/run-info.txt"
printf 'gc_skip_text_last=%s\n' "$gc_skip" >> "$out_host/run-info.txt"
fi
if [[ "$nodes" == 2 ]]; then
  signature="$revision $job $batch $steps $warmup $gateup $fast_zero $gc_skip"
  if [[ "$rank" == 0 ]]; then
    printf '%s\n' "$signature" > "$out_host/pair.tmp"
    mv "$out_host/pair.tmp" "$out_host/pair"
  else
    for _ in $(seq 1 90); do [[ ! -s "$out_host/pair" ]] || break; sleep 2; done
    [[ -s "$out_host/pair" && "$(< "$out_host/pair")" == "$signature" ]] || die 'Rank 0 missing or configuration differs.'
    (set -o noclobber; : > "$out_host/rank1.claim") || die 'Rank 1 already used; choose a new run ID.'
  fi
fi
printf 'Results: %s\n' "$out_host"

# Explicitly pin CPU-only shape sampling and common training switches.
in_box() {
  docker exec -i "$box" bash --noprofile --norc -e -s -- "$@" <<'INSIDE'
source /work/zhaobo/bootstrap/enter-env.sh
set -eo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD:$PWD/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
if [[ $1 == python ]]; then
  shift
  exec env DEVICE=cpu python "$@"
fi
profile=$1 stage=$2 steps=$3 warmup=$4 batch=$5 ip=$6 out=$7 model=$8 sd=$9 data=${10} rgb=${11} helper=${12} gateup=${13} fast_zero=${14}
nodes=${15} rank=${16} master=${17} job=${18}
gc_skip=${19:-0}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=enp39s0f3 HCCL_IF_IP="$ip" HCCL_CONNECT_TIMEOUT=120
export ROSS_COUNT_FLOPS=0 ROSS_STEP_TIMING=1 ROSS_NPU_RMSNORM=1 ROSS_NPU_ROPE=1
export ROSS_FUSED_GATEUP=0 ROSS_FUSED_QKV=0 ROSS_GC_SKIP_LAST_LAYERS="$gc_skip"
if [[ $profile == fast ]]; then
  attn=fa2; ds=zero$fast_zero.json; pin=true; workers=2
  if [[ $fast_zero == 2 ]]; then
    ZB_REAL_TORCHRUN=$(command -v torchrun)
    export ZB_REAL_TORCHRUN ZB_ZERO2_COMPAT=${helper%/*}/npu_zero2_compat.py
    export PATH="${helper%/*}/mfu-fast-bin:$PATH"
  fi
  export ROSS_FUSED_GATEUP="$gateup"
else
  attn=sdpa; ds=zero3_offload.json; pin=false; workers=0; fast_zero=3
fi
extra=()
if [[ "$job" == profile ]]; then
  extra+=(--npu_profiling --npu_profiler_level 1)
  out="$out/node-rank-$rank"
  export ROSS_STEP_TIMING=0
fi
exec bash scripts/scripts_train/train_qwen_vla_navsim.sh \
  --nnodes "$nodes" --node_rank "$rank" --master_addr "$master" --master_port 29555 --ngpus 8 \
  --model_name_or_path "$model" --sd_model_path "$sd" --data_path "$data" --data_root "$rgb" \
  --device npu --fp bf16 --attn_type "$attn" --zero_stage "$fast_zero" --deepspeed_config "$PWD/scripts/sft/$ds" \
  --pin_memory "$pin" --num_workers "$workers" --batch_size "$batch" --grad_accum 1 \
  --max_pixels 903168 --min_pixels 12544 --model_max_length 4096 --seed 42 \
  --max_steps "$steps" --warmup_steps "$warmup" --logging_steps 1 --save_model false \
  --report_to none --output_dir "$out/$stage-$profile" --exp_name navsim "${extra[@]}"
INSIDE
}

if [[ "$nodes" == 2 ]]; then
  in_box fast train "$steps" "$warmup" "$batch" "$ip" "$out_box" "$model" "$sd" "$data" "$rgb" "$helper" "$gateup" "$fast_zero" "$nodes" "$rank" "$master" "$job" "$gc_skip" |& tee "$out_host/train-$node.log"
  if [[ "$rank" == 0 && "$job" == measure ]]; then
    cp "$out_host/train-$node.log" "$out_host/train-fast.log"
    in_box python "$helper" shapes "$model" "$data" "$rgb" "$out_box/shapes.json" |& tee "$out_host/shapes.log"
    in_box python "$helper" report "$out_box" "$batch" fast --steps "$steps" --warmup "$warmup" --devices 16 |& tee "$out_host/summary.txt"
  fi
  exit 0
fi
(cd "$release" && ZB_STRICT_RELEASE=1 ZB_NODE="$node" \
  bash start-training.sh check "$run_id" safe "$batch" qwen3vl 1) |& tee "$out_host/hccl.log"
in_box python "$helper" shapes "$model" "$data" "$rgb" "$out_box/shapes.json" |& tee "$out_host/shapes.log"

completed=()
for profile in "${profiles[@]}"; do
  if [[ "$smoke" == 1 ]]; then
    assert_idle
    in_box "$profile" smoke 3 0 "$batch" "$ip" "$out_box" "$model" "$sd" "$data" "$rgb" "$helper" "$gateup" "$fast_zero" 1 0 "$ip" measure "$gc_skip" |& tee "$out_host/smoke-$profile.log"
  fi
  assert_idle
  in_box "$profile" train "$steps" "$warmup" "$batch" "$ip" "$out_box" "$model" "$sd" "$data" "$rgb" "$helper" "$gateup" "$fast_zero" 1 0 "$ip" measure "$gc_skip" |& tee "$out_host/train-$profile.log"
  completed+=("$profile")
  in_box python "$helper" report "$out_box" "$batch" "${completed[@]}" --steps "$steps" --warmup "$warmup" |& tee "$out_host/summary.txt"
done
printf 'Summary: %s/summary.txt\n' "$out_host"
