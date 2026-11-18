#!/usr/bin/env bash
# Qwen3-VL + ROSS performance baseline on one reserved 8-NPU A5 host.
set -euo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ -n ${ZB_NODE:-} ]]; then
  node=$ZB_NODE
else
  case $(hostname -s) in
    liteserver-a5-00001) node=a51 ;;
    liteserver-a5-00002) node=a52 ;;
    liteserver-a5-00003) node=a53 ;;
    liteserver-a5-00004) node=a54 ;;
    *) die 'Set ZB_NODE or run this on a preconfigured A5 host.' ;;
  esac
fi
[[ "$node" =~ ^[A-Za-z][A-Za-z0-9_-]{0,63}$ ]] ||
  die "Invalid ZB_NODE: $node"
[[ $# -le 1 && ( $# -eq 0 || $1 == --dry-run ) ]] || die 'Usage: bash benchmark-mfu.sh [--dry-run]'
shared=/mnt/sfs_turbo/zhaobo/drivevla-w0
host=/data1/zhaobo/drivevla-w0-$node
box=zhaobo-drivevla-$node
release=$shared/artifacts/code/a55-a56
release=${ZB_RELEASE_DIR:-"$release"}
repo=$host/code/DriveVLA-W0
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ "$here" == "$shared/"* ]] || die 'Run the copy on the shared drive.'
helper=/work/zhaobo/${here#"$shared/"}/benchmark_mfu.py
nodes=1 rank=0 job=${ZB_JOB:-measure}
node_list=
[[ "$job" == measure || "$job" == audit || "$job" == profile || "$job" == smoke ]] || die 'Invalid ZB_JOB.'
multi_id=
if [[ -n ${ZB_PAIR_RUN:-} && -n ${ZB_MULTI_RUN:-} ]]; then
  die 'Set only one of ZB_PAIR_RUN or ZB_MULTI_RUN.'
fi
if [[ -n ${ZB_PAIR_RUN:-} ]]; then
  [[ "$ZB_PAIR_RUN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || die 'Invalid ZB_PAIR_RUN.'
  multi_id=$ZB_PAIR_RUN
  nodes=2
  node_list=${ZB_NODE_LIST:-a53,a54}
  export ZB_MFU_PROFILE=fast ZB_MFU_SMOKE=0
elif [[ -n ${ZB_MULTI_RUN:-} ]]; then
  [[ "$ZB_MULTI_RUN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || die 'Invalid ZB_MULTI_RUN.'
  multi_id=$ZB_MULTI_RUN
  nodes=${ZB_SCALE_NODES:-4}
  [[ "$nodes" == 2 || "$nodes" == 4 ]] || die 'ZB_SCALE_NODES must be 2 or 4 for a multi-node run.'
  if [[ -n ${ZB_NODE_LIST:-} ]]; then
    node_list=$ZB_NODE_LIST
  elif [[ "$nodes" == 2 ]]; then
    node_list=a51,a52
  else
    node_list=a51,a52,a53,a54
  fi
  export ZB_MFU_PROFILE=fast ZB_MFU_SMOKE=0
fi
if [[ "$nodes" -gt 1 ]]; then
  IFS=',' read -r -a scale_nodes <<< "$node_list"
  [[ ${#scale_nodes[@]} -eq "$nodes" ]] || die "ZB_NODE_LIST must contain exactly $nodes nodes."
  declare -A seen_nodes=()
  rank=-1
  for index in "${!scale_nodes[@]}"; do
    [[ ${scale_nodes[$index]} =~ ^[A-Za-z][A-Za-z0-9_-]{0,63}$ ]] ||
      die "Invalid node in ZB_NODE_LIST: ${scale_nodes[$index]}"
    [[ -z ${seen_nodes[${scale_nodes[$index]}]:-} ]] ||
      die "Duplicate node in ZB_NODE_LIST: ${scale_nodes[$index]}"
    seen_nodes[${scale_nodes[$index]}]=1
    [[ ${scale_nodes[$index]} != "$node" ]] || rank=$index
  done
  [[ "$rank" -ge 0 ]] || die "$node is not part of ZB_NODE_LIST=$node_list"
fi
if [[ -n ${ZB_SINGLE_RUN:-} ]]; then
  [[ "$nodes" == 1 && ( "$job" == measure || "$job" == audit || "$job" == profile ) ]] ||
    die 'ZB_SINGLE_RUN is for single-node measure, audit or profile.'
  [[ "$ZB_SINGLE_RUN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$ ]] || die 'Invalid ZB_SINGLE_RUN.'
  export ZB_MFU_PROFILE=fast ZB_MFU_SMOKE=0
fi
[[ "$job" != audit || "$nodes" == 2 || -n ${ZB_SINGLE_RUN:-} ]] || die 'Projection audit needs a named run.'
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
fa2_cache=${ZB_FA2_UNPAD_CACHE:-0}
fa2_markers=${ZB_FA2_UNPAD_MARKERS:-0}
fa2_stats=${ZB_FA2_UNPAD_STATS:-0}
profile_full=${ZB_PROFILE_FULL:-0}
zero2_bucket=${ZB_ZERO2_BUCKET_SIZE:-auto}
zero2_overlap=${ZB_ZERO2_OVERLAP_COMM:-1}
hccl_deterministic=${ZB_HCCL_DETERMINISTIC:-1}
hccl_deterministic_value=FALSE
[[ "$hccl_deterministic" == 0 ]] || hccl_deterministic_value=TRUE
[[ "$gc_skip" =~ ^(0|[1-9][0-9]?)$ ]] || die 'Invalid ZB_GC_SKIP_TEXT_LAST.'
[[ "$fa2_cache" =~ ^[01]$ && "$fa2_markers" =~ ^[01]$ && "$fa2_stats" =~ ^[01]$ ]] ||
  die 'FA2 switches must be 0 or 1.'
[[ "$profile_full" == 0 || ( "$profile_full" == 1 && "$job" == profile && "$steps" == 50 ) ]] ||
  die 'ZB_PROFILE_FULL=1 requires the 50-step profile job.'
[[ "$zero2_bucket" == auto || "$zero2_bucket" =~ ^[1-9][0-9]*$ ]] ||
  die 'ZB_ZERO2_BUCKET_SIZE must be auto or a positive integer element count.'
[[ "$zero2_overlap" =~ ^[01]$ && "$hccl_deterministic" =~ ^[01]$ ]] ||
  die 'ZB_ZERO2_OVERLAP_COMM and ZB_HCCL_DETERMINISTIC must be 0 or 1.'
if [[ "$fa2_cache" == 1 || "$fa2_markers" == 1 || "$fa2_stats" == 1 ]]; then
  [[ "${profiles[*]}" == fast ]] || die 'FA2 experiments require the fast profile.'
fi
[[ "$steps" =~ ^[1-9][0-9]{0,2}$ && "$warmup" =~ ^(0|[1-9][0-9]{0,2})$ && "$steps" -gt "$warmup" ]] ||
  die 'Set 0 <= ZB_MFU_WARMUP < ZB_MFU_STEPS <= 999.'
[[ "$job" != audit || "$warmup" -ge 1 ]] || die 'Projection audit requires at least one warmup step.'
[[ -z ${ZB_SINGLE_RUN:-} || "$warmup" -ge 1 ]] || die 'Scaling baseline requires at least one warmup step.'
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
train_repo=$repo
box_code=/work/zhaobo/code/DriveVLA-W0
snapshot=base
if [[ -d "$here/checkout" ]]; then
  train_repo=$here/checkout
  [[ -s "$train_repo/.mfu-base-revision" && "$(< "$train_repo/.mfu-base-revision")" == "$revision" ]] ||
    die 'Benchmark snapshot is not based on the installed revision.'
  (cd "$train_repo" && sha256sum -c .mfu-sha256 --status) || die 'Benchmark snapshot checksums differ.'
  snapshot=$(sha256sum "$train_repo/.mfu-sha256")
  snapshot=${snapshot%% *}
  box_code=/work/zhaobo/${train_repo#"$shared/"}
fi
grep -q ' READY$' "$host/manifests/prepare-environment.status" || die 'Training environment not ready.'
[[ $(docker inspect -f '{{.State.Running}} {{.HostConfig.NetworkMode}}' "$box") == 'true host' ]] ||
  die 'Training container must be running with host networking.'
shared_memory_bytes=$(docker inspect -f '{{.HostConfig.ShmSize}}' "$box")
[[ "$shared_memory_bytes" =~ ^[0-9]+$ && "$shared_memory_bytes" -ge 34359738368 ]] ||
  die "Training container needs at least 32 GiB /dev/shm (got $shared_memory_bytes bytes)."
ip=$(ip -o -4 addr show dev enp39s0f3 | awk '{sub(/\/.+$/, "", $4); print $4; exit}')
[[ -n "$ip" ]] || die 'No IPv4 on enp39s0f3.'
master=$ip
if [[ "$nodes" -gt 1 ]]; then
  node_ip() {
    case "$1" in
      a51) printf '%s' 10.0.16.12 ;;
      a52) printf '%s' 10.0.16.75 ;;
      a53) printf '%s' 10.0.16.231 ;;
      a54) printf '%s' 10.0.16.118 ;;
      *)
        { local key="ZB_NODE_IP_${1^^}"; key=${key//-/_}; }
        [[ -n ${!key:-} ]] || return 1
        printf '%s' "${!key}"
        ;;
    esac
  }
  master=$(node_ip "${scale_nodes[0]}")
  expected_ip=$(node_ip "$node")
  [[ "$ip" == "$expected_ip" ]] || die "$node rendezvous IP changed: expected $expected_ip, got $ip"
fi
if [[ "$job" == profile ]]; then
  [[ "$steps" -ge 30 ]] || die 'Profiler requires at least 30 steps.'
fi
printf 'nodes=%s rank=%s master=%s:29555 phase=%s\n' "$nodes" "$rank" "$master" "$job"
printf 'gc_skip_text_last=%s fa2_unpad_cache=%s fa2_unpad_markers=%s zero2_bucket=%s zero2_overlap=%s hccl_deterministic=%s\n' \
  "$gc_skip" "$fa2_cache" "$fa2_markers" "$zero2_bucket" "$zero2_overlap" "$hccl_deterministic"
docker exec "$box" bash -e -c '
  test -r "$1" && test -s "$2/config.json" && test -d "$3/unet" &&
  test -d "$3/vae" && test -s "$4" && test -d "$5"
' _ "$helper" "$model" "$sd" "$data" "$rgb" || die 'Benchmark helper, model or dataset missing inside the container.'
docker exec "$box" bash --noprofile --norc -e -c '
  source /work/zhaobo/bootstrap/enter-env.sh
  export PYTHONPATH="$PWD/reference/transformers/src:$PYTHONPATH"
  python -c "import importlib.util as u; missing = [p for p in (\"torch\", \"torch_npu\", \"transformers\", \"tokenizers\", \"deepspeed\", \"diffusers\") if not u.find_spec(p)]; raise SystemExit(\"Missing packages: \" + \", \".join(missing) if missing else 0)"
' || die 'Recreated container may need the project environment preparation again.'
status=$(cd "$release" && ZB_NODE="$node" bash start-training.sh status)
printf '%s\n' "$status"
[[ "$status" == *'sync=ok'* ]] || die 'Shared release files are out of sync.'
if [[ -n ${ZB_SINGLE_RUN:-} ]] ||
   [[ "$nodes" -gt 1 && ( "$job" == measure || "$job" == audit ) ]]; then
  [[ -s "$train_repo/utils/projection_mfu.py" ]] &&
    grep -q ROSS_MFU_STEP_WALL "$train_repo/utils/train_qwen_vla.py" ||
    die 'Sync the projection audit and wall timing code before running this launcher.'
fi
if [[ "$fa2_cache" == 1 || "$fa2_markers" == 1 || "$fa2_stats" == 1 ]]; then
  [[ -s "$train_repo/utils/fa2_unpad_cache.py" ]] &&
    grep -q ROSS_FA2_UNPAD_CACHE "$train_repo/utils/train_qwen_vla.py" ||
    die 'The selected checkout does not have the FA2 unpadding experiment.'
fi
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
if [[ -n ${ZB_SINGLE_RUN:-} ]]; then run_id=mfu-$node-$ZB_SINGLE_RUN-$job; fi
if [[ "$nodes" -gt 1 ]]; then
  node_label=$(IFS=-; printf '%s' "${scale_nodes[*]}")
  run_id=mfu-$node_label-$multi_id-$job
fi
out=artifacts/runs/mfu/$run_id
out_host=$shared/$out
out_box=/work/zhaobo/$out
assert_idle
mkdir -p "$shared/artifacts/runs/mfu"
if [[ "$rank" == 0 ]]; then
mkdir "$out_host"
printf 'node=%s\nrevision=%s\nmbs=%s\ngbs=%s\nsteps=%s\nwarmup=%s\nfast_gateup=%s\nfast_zero=%s\npeak_flops_assumed=486000000000000\n' \
  "$node" "$revision" "$batch" "$((batch * 8 * nodes))" "$steps" "$warmup" "$gateup" "$fast_zero" > "$out_host/run-info.txt"
printf 'node_list=%s\nnode_rank=%s\nshared_memory_bytes=%s\n' \
  "$node_list" "$rank" "$shared_memory_bytes" >> "$out_host/run-info.txt"
printf 'gc_skip_text_last=%s\n' "$gc_skip" >> "$out_host/run-info.txt"
printf 'fa2_unpad_cache=%s\nfa2_unpad_markers=%s\nfa2_unpad_stats=%s\n' "$fa2_cache" "$fa2_markers" "$fa2_stats" >> "$out_host/run-info.txt"
printf 'zero2_bucket_size=%s\nzero2_overlap_comm=%s\nhccl_deterministic=%s\n' \
  "$zero2_bucket" "$zero2_overlap" "$hccl_deterministic" >> "$out_host/run-info.txt"
if [[ " ${profiles[*]} " == *' fast '* && "$fast_zero" == 2 ]]; then
  [[ -s "$train_repo/utils/render_deepspeed_config.py" ]] ||
    die 'The selected checkout does not have the DeepSpeed config renderer.'
  python3 "$train_repo/utils/render_deepspeed_config.py" \
    "$train_repo/scripts/sft/zero2.json" "$out_host/deepspeed-zero2.json" \
    --reduce-bucket-size "$zero2_bucket" --overlap-comm "$zero2_overlap"
fi
scaling_workload="$revision $snapshot ${profiles[*]} $batch $steps $warmup $gateup $fast_zero $gc_skip $fa2_markers $fa2_stats $model $sd $data $rgb seed42 fa2_cache=$fa2_cache zero2_bucket=$zero2_bucket zero2_overlap=$zero2_overlap hccl_deterministic=$hccl_deterministic"
printf 'job=%s\nworld_size=%s\nworkload=%s\nscaling_workload=%s\n' "$job" "$((8 * nodes))" \
  "$nodes $scaling_workload" "$scaling_workload" \
  >> "$out_host/run-info.txt"
fi
if [[ "$nodes" -gt 1 ]]; then
  signature="$revision $snapshot $nodes $node_list $job $batch $steps $warmup $gateup $fast_zero $gc_skip $fa2_cache $fa2_markers $fa2_stats $profile_full $zero2_bucket $zero2_overlap $hccl_deterministic"
  if [[ "$rank" == 0 ]]; then
    printf '%s\n' "$signature" > "$out_host/group.tmp"
    mv "$out_host/group.tmp" "$out_host/group"
    for peer_rank in $(seq 1 $((nodes - 1))); do
      for _ in $(seq 1 90); do [[ ! -e "$out_host/rank$peer_rank.claim" ]] || break; sleep 2; done
      [[ -e "$out_host/rank$peer_rank.claim" ]] ||
        die "Rank $peer_rank did not finish preflight within 180 seconds."
    done
  else
    for _ in $(seq 1 90); do [[ ! -s "$out_host/group" ]] || break; sleep 2; done
    [[ -s "$out_host/group" && "$(< "$out_host/group")" == "$signature" ]] ||
      die 'Rank 0 missing or configuration differs.'
    (set -o noclobber; : > "$out_host/rank$rank.claim") ||
      die "Rank $rank already used; choose a new run ID."
  fi
fi
printf 'Results: %s\n' "$out_host"

# Explicitly pin CPU-only shape sampling and common training switches.
in_box() {
  docker exec -i -e ZB_BENCH_CODE="$box_code" -e ROSS_FA2_UNPAD_CACHE="$fa2_cache" \
    -e ROSS_FA2_UNPAD_MARKERS="$fa2_markers" -e ROSS_FA2_UNPAD_STATS="$fa2_stats" -e ZB_PROFILE_FULL="$profile_full" \
    -e ZB_ZERO2_CONFIG="$out_box/deepspeed-zero2.json" \
    -e HCCL_DETERMINISTIC="$hccl_deterministic_value" \
    "$box" bash --noprofile --norc -e -s -- "$@" <<'INSIDE'
source /work/zhaobo/bootstrap/enter-env.sh
set -eo pipefail
cd "$ZB_BENCH_CODE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD:$PWD/reference/transformers/src:$PWD/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
if [[ $1 == python ]]; then
  shift
  exec env DEVICE=cpu python "$@"
fi
profile=$1 stage=$2 steps=$3 warmup=$4 batch=$5 ip=$6 out=$7 model=$8 sd=$9 data=${10} rgb=${11} helper=${12} gateup=${13} fast_zero=${14}
nodes=${15} rank=${16} master=${17} job=${18}
gc_skip=${19:-0}
single_scaling=${20:-0}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=enp39s0f3 HCCL_IF_IP="$ip" HCCL_CONNECT_TIMEOUT=120
export ROSS_COUNT_FLOPS=0 ROSS_STEP_TIMING=1 ROSS_NPU_RMSNORM=1 ROSS_NPU_ROPE=1
export ROSS_PROJECTION_AUDIT=0 ROSS_MFU_STEP_WALL=0
[[ "$job" != audit ]] || export ROSS_PROJECTION_AUDIT=1
if [[ "$single_scaling" == 1 ]] ||
   [[ "$nodes" -gt 1 && ( "$job" == measure || "$job" == audit ) ]]; then
  export ROSS_MFU_STEP_WALL=1
fi
export ROSS_FUSED_GATEUP=0 ROSS_FUSED_QKV=0 ROSS_GC_SKIP_LAST_LAYERS="$gc_skip"
if [[ $profile == fast ]]; then
  attn=fa2; ds=zero$fast_zero.json; pin=true; workers=2
  if [[ $fast_zero == 2 ]]; then
    [[ -s "$ZB_ZERO2_CONFIG" ]] || { printf 'ERROR: generated ZeRO-2 config missing: %s\n' "$ZB_ZERO2_CONFIG" >&2; exit 1; }
    ds_path=$ZB_ZERO2_CONFIG
    ZB_REAL_TORCHRUN=$(command -v torchrun)
    export ZB_REAL_TORCHRUN ZB_ZERO2_COMPAT=${helper%/*}/npu_zero2_compat.py
    export PATH="${helper%/*}/mfu-fast-bin:$PATH"
  fi
  export ROSS_FUSED_GATEUP="$gateup"
else
  attn=sdpa; ds=zero3_offload.json; pin=false; workers=0; fast_zero=3
fi
ds_path=${ds_path:-$PWD/scripts/sft/$ds}
extra=()
if [[ "$job" == profile ]]; then
  extra+=(--npu_profiling --npu_profiler_level 1)
  out="$out/node-rank-$rank"
  export ROSS_STEP_TIMING=0
  if [[ "$ZB_PROFILE_FULL" == 1 ]]; then export ROSS_PROFILE_FULL_STEPS="$steps"; fi
fi
exec bash scripts/scripts_train/train_qwen_vla_navsim.sh \
  --nnodes "$nodes" --node_rank "$rank" --master_addr "$master" --master_port 29555 --ngpus 8 \
  --model_name_or_path "$model" --sd_model_path "$sd" --data_path "$data" --data_root "$rgb" \
  --device npu --fp bf16 --attn_type "$attn" --zero_stage "$fast_zero" --deepspeed_config "$ds_path" \
  --pin_memory "$pin" --num_workers "$workers" --batch_size "$batch" --grad_accum 1 \
  --max_pixels 903168 --min_pixels 12544 --model_max_length 4096 --seed 42 \
  --max_steps "$steps" --warmup_steps "$warmup" --logging_steps 1 --save_model false \
  --report_to none --output_dir "$out/$stage-$profile" --exp_name navsim "${extra[@]}"
INSIDE
}

if [[ "$nodes" -gt 1 ]]; then
  multi_stage=train
  [[ "$job" != profile ]] || multi_stage=profile
  if in_box fast "$multi_stage" "$steps" "$warmup" "$batch" "$ip" "$out_box" "$model" "$sd" "$data" "$rgb" "$helper" "$gateup" "$fast_zero" "$nodes" "$rank" "$master" "$job" "$gc_skip" |& tee "$out_host/train-$node.log"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$out_host/train-$node.exit"
  [[ "$rc" == 0 ]] || exit "$rc"
  if [[ "$rank" == 0 && "$job" == measure ]]; then
    cp "$out_host/train-$node.log" "$out_host/train-fast.log"
    in_box python "$helper" shapes "$model" "$data" "$rgb" "$out_box/shapes.json" |& tee "$out_host/shapes.log"
    in_box python "$helper" report "$out_box" "$batch" fast --steps "$steps" --warmup "$warmup" --devices "$((8 * nodes))" |& tee "$out_host/summary.txt"
  fi
  exit 0
fi
(cd "$release" && ZB_STRICT_RELEASE=1 ZB_NODE="$node" \
  bash start-training.sh check "$run_id" safe "$batch" qwen3vl 1) |& tee "$out_host/hccl.log"
if [[ "$job" == measure || "$job" == smoke ]]; then
  in_box python "$helper" shapes "$model" "$data" "$rgb" "$out_box/shapes.json" |& tee "$out_host/shapes.log"
fi

completed=()
single_scaling=0
[[ -z ${ZB_SINGLE_RUN:-} ]] || single_scaling=1
for profile in "${profiles[@]}"; do
  if [[ "$smoke" == 1 ]]; then
    assert_idle
    in_box "$profile" smoke 3 0 "$batch" "$ip" "$out_box" "$model" "$sd" "$data" "$rgb" "$helper" "$gateup" "$fast_zero" 1 0 "$ip" measure "$gc_skip" |& tee "$out_host/smoke-$profile.log"
  fi
  assert_idle
  if in_box "$profile" train "$steps" "$warmup" "$batch" "$ip" "$out_box" "$model" "$sd" "$data" "$rgb" "$helper" "$gateup" "$fast_zero" 1 0 "$ip" "$job" "$gc_skip" "$single_scaling" |& tee "$out_host/train-$profile.log"; then
    rc=0
  else
    rc=$?
  fi
  if [[ "$single_scaling" == 1 ]]; then
    printf '%s\n' "$rc" > "$out_host/train-$node.exit"
  fi
  [[ "$rc" == 0 ]] || exit "$rc"
  completed+=("$profile")
  if [[ "$job" == measure || "$job" == smoke ]]; then
    in_box python "$helper" report "$out_box" "$batch" "${completed[@]}" --steps "$steps" --warmup "$warmup" |& tee "$out_host/summary.txt"
  fi
done
printf 'Summary: %s/summary.txt\n' "$out_host"
