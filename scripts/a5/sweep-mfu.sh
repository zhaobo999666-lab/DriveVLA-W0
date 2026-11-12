#!/usr/bin/env bash
# Tune the repaired fast profile after a full batch-2 run has passed.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base=/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu
if (( $# == 0 )); then set -- 8 16 24 27 32; fi
for batch in "$@"; do
  [[ "$batch" =~ ^[1-9][0-9]{0,2}$ && "$batch" -ge 2 ]] || { echo "Invalid batch: $batch" >&2; exit 2; }
done
sweep=$(mktemp -d "$base/sweep-$(hostname -s)-XXXXXX")
printf 'Sweep: %s\n' "$sweep"
printf 'mbs\tuseful_mfu_pct\trun_directory\n' > "$sweep/results.tsv"
for batch in "$@"; do
  log="$sweep/b$batch.log"
  printf '\n=== Pilot: MBS %s ===\n' "$batch"
  if ZB_MFU_PROFILE=fast ZB_MBS="$batch" ZB_FUSED_GATEUP=1 \
    ZB_MFU_SMOKE=0 ZB_MFU_STEPS=12 ZB_MFU_WARMUP=5 bash "$here/benchmark-mfu.sh" |& tee "$log"; then
    run=$(sed -n 's/^Results: //p' "$log" | tail -1)
    mfu=$(sed -nE 's/^fast: .*useful MFU=([0-9.]+)%.*/\1/p' "$run/summary.txt")
    [[ -n "$mfu" ]] || { echo "Missing MFU at batch $batch" >&2; exit 1; }
    printf '%s\t%s\t%s\n' "$batch" "$mfu" "$run" | tee -a "$sweep/results.tsv"
  elif grep -Eqi 'out of memory|ACL_ERROR_RT_MEMORY_ALLOCATION' "$log"; then
    printf 'MBS %s did not fit; skipping larger batches.\n' "$batch" | tee "$sweep/stop.txt"
    break
  else
    echo "Batch $batch failed; see $log" >&2
    exit 1
  fi
done
best=$(awk -F '\t' 'NR>1 && $2+0>best {best=$2+0; b=$1} END {print b}' "$sweep/results.tsv")
[[ -n "$best" ]] || { echo 'No successful pilot run.' >&2; exit 1; }
printf '\n=== Confirming MBS %s over 50 steps ===\n' "$best"
ZB_MFU_PROFILE=fast ZB_MBS="$best" ZB_FUSED_GATEUP=1 ZB_MFU_SMOKE=0 \
  ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10 bash "$here/benchmark-mfu.sh" |& tee "$sweep/confirmation.log"
printf 'Sweep results: %s/results.tsv\n' "$sweep"
