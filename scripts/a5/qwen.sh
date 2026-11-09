#!/usr/bin/env bash
# Qwen2.5-VL + ROSS on the existing A5/CANN container.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${1:-check}
if [ "$#" -gt 0 ]; then shift; fi
VENV_DIR=${VENV_DIR:-/data1/drivevla-w0-codex/qwen-py312}
BASE_PYTHON=${BASE_PYTHON:-/opt/cosmos3_py312/bin/python}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
MODEL_PATH=${MODEL_PATH:-/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct}
SD_MODEL_PATH=${SD_MODEL_PATH:-/mnt/sfs_turbo/models/stable-diffusion-v1-5}

case "$MODE" in
  setup|check|smoke|train) ;;
  *) echo "Usage: bash scripts/a5/qwen.sh {setup|check|smoke|train} [training flags]" >&2; exit 2 ;;
esac
[ -f "$CANN_ENV" ] || { echo "Missing CANN environment: $CANN_ENV" >&2; exit 1; }
# Vendor environment scripts can reference unset variables.
set +u
source "$CANN_ENV"
set -u
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"

if [ "$MODE" = setup ]; then
  [ -x "$BASE_PYTHON" ] || { echo "Missing base interpreter: $BASE_PYTHON" >&2; exit 1; }
  # Reuse the image's A5-matched torch/torch_npu without changing its environment.
  "$BASE_PYTHON" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
  if [ -e "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Existing venv is broken: $VENV_DIR. Choose a new VENV_DIR." >&2
    exit 1
  fi
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT/reference/transformers" \
    'diffusers==0.39.0' 'huggingface_hub==0.36.2' \
    accelerate tensorboard scipy pillow pyyaml tiktoken einops safetensors sentencepiece
  "$VENV_DIR/bin/python" -c 'import deepspeed; print("DeepSpeed", deepspeed.__version__)'
  # A system torchrun has a shebang pointing to the base interpreter. Force the
  # existing project launcher to use this venv, including its local Transformers.
  cat > "$VENV_DIR/bin/torchrun" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/python" -m torch.distributed.run "$@"
SH
  chmod +x "$VENV_DIR/bin/torchrun"
fi

[ -x "$VENV_DIR/bin/python" ] || {
  echo "Run setup first inside the A5 container: bash scripts/a5/qwen.sh setup" >&2
  exit 1
}
export PATH="$VENV_DIR/bin:$PATH"
export DEVICE=npu

python - <<'PY'
import sys
import torch, torch_npu, transformers, deepspeed, diffusers
from transformers import Qwen2_5_VLForConditionalGeneration
from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_ross import Qwen2_5_VLForConditionalGenerationROSS
print('Python:', sys.executable)
print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__)
print('Transformers:', transformers.__version__, transformers.__file__)
print('DeepSpeed:', deepspeed.__version__, 'diffusers:', diffusers.__version__)
print('Qwen and ROSS imports: OK (no NPU tensor allocated)')
PY
# Parse the actual training entry without constructing or training a model.
DEVICE=cpu python utils/train_qwen_vla.py --help >/dev/null

if [ "$MODE" = setup ] || [ "$MODE" = check ]; then
  echo "Environment and argument parser checks passed. No training was started."
  exit 0
fi

# All eight devices on A55 were occupied when this launcher was prepared.
# Refuse a conflicting launch rather than stopping or joining an existing job.
npu-smi info | python -c '
import re,sys
s=sys.stdin.read()
busy=re.findall(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|",s,re.M)
if busy:
    sys.exit("NPU processes already present; wait for the existing job to release devices. PIDs: "+", ".join(busy))
'
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_LAUNCH_BLOCKING=0
export INF_NAN_MODE_ENABLE=1
export ROSS_NPU_RMSNORM=${ROSS_NPU_RMSNORM:-0}
export ROSS_NPU_ROPE=${ROSS_NPU_ROPE:-0}
export ROSS_FUSED_GATEUP=${ROSS_FUSED_GATEUP:-0}
export ROSS_FUSED_SWIGLU=${ROSS_FUSED_SWIGLU:-0}
export ROSS_FUSED_ADD_RMSNORM=${ROSS_FUSED_ADD_RMSNORM:-0}

COMMON=(
  --model_name_or_path "$MODEL_PATH" --sd_model_path "$SD_MODEL_PATH"
  --device npu --attn_type sdpa --fp bf16
  --batch_size 2 --num_workers 0 --pin_memory false
  --deepspeed_config "$ROOT/scripts/sft/zero3_offload.json"
  --max_pixels 150528 --min_pixels 12544 --use_previous_actions false
  --master_port 29555 --logging_steps 1 --report_to none
)
if [ "$MODE" = smoke ]; then
  SYNTH_ROOT=${SYNTH_ROOT:-$ROOT/logs/a5_synth_qwen}
  python tools/synth/synth_qwen_vla_data.py --out_root "$SYNTH_ROOT" --num_scenes 64
  bash scripts/scripts_train/train_qwen_vla_navsim.sh "${COMMON[@]}" \
    --ngpus 1 --max_steps 3 --warmup_steps 0 --save_model false \
    --data_path "$SYNTH_ROOT/synth_navsim.pkl" --data_root "$SYNTH_ROOT/images" \
    --output_dir "$ROOT/logs/a5_qwen_smoke_$(date +%Y%m%d_%H%M%S)" "$@"
else
  : "${DATA_PATH:?Set DATA_PATH to the real NAVSIM pickle}"
  : "${DATA_ROOT:?Set DATA_ROOT to the matching RGB camera root}"
  bash scripts/scripts_train/train_qwen_vla_navsim.sh "${COMMON[@]}" \
    --ngpus 8 --max_steps 4000 --warmup_steps 50 --save_model true \
    --data_path "$DATA_PATH" --data_root "$DATA_ROOT" \
    --output_dir "$ROOT/logs/a5_qwen_train_$(date +%Y%m%d_%H%M%S)" "$@"
fi
