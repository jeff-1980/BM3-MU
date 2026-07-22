#!/bin/bash
# experiments/exp_e3_lobo_leakfree/wait_e2_then_run_e3.sh
#
# Poll every 5 minutes for E2 (results/e2_cnn_ln_20260709-1723/) to fully land
# (7 arm/config dirs, each with a summary.json — 6 CWRU SNR/clean points +
# 1 XJTU cross point, matching experiments/exp_p2_cnn1d_ln/run_cnn_ln.sh's
# SNR_GRID and XJTU section). Once all 7 are present, locate and launch
# run_lobo_leakfree.sh, tee'd to a timestamped log under results/.
#
# Intended usage: run inside tmux session "e3" (this file does not create
# the session itself).

set -u
cd "$(dirname "$0")/../.."

E2_ROOT="results/e2_cnn_ln_20260709-1723"
E2_ARMS=(
  "cwru_cnn1d_ln_snrclean"
  "cwru_cnn1d_ln_snr0"
  "cwru_cnn1d_ln_snrm2"
  "cwru_cnn1d_ln_snrm4"
  "cwru_cnn1d_ln_snrm6"
  "cwru_cnn1d_ln_snrm8"
  "xjtu_cross_cnn1d_ln"
)

echo "$(date '+%F %T')  waiting for E2 (${#E2_ARMS[@]} arms under ${E2_ROOT}) — polling every 5 min"

while true; do
  missing=()
  for arm in "${E2_ARMS[@]}"; do
    if [ ! -f "${E2_ROOT}/${arm}/summary.json" ]; then
      missing+=("$arm")
    fi
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    echo "$(date '+%F %T')  E2 complete: all ${#E2_ARMS[@]} summary.json present."
    break
  fi
  echo "$(date '+%F %T')  E2 not complete yet — missing: ${missing[*]} — sleeping 5min"
  sleep 300
done

RUN_SCRIPT=$(find "$(dirname "$0")/../.." -iname "run_lobo_leakfree.sh" | head -1)
if [ -z "$RUN_SCRIPT" ]; then
  echo "$(date '+%F %T')  ERROR: run_lobo_leakfree.sh not found, aborting."
  exit 1
fi
echo "$(date '+%F %T')  found run script: ${RUN_SCRIPT}"

LOG="results/e3_lobo_leakfree_$(date +%Y%m%d-%H%M%S).log"
echo "$(date '+%F %T')  launching ${RUN_SCRIPT}, log -> ${LOG}"
bash "$RUN_SCRIPT" 2>&1 | tee "$LOG"
echo "$(date '+%F %T')  wait_e2_then_run_e3.sh finished"
