#!/usr/bin/env bash
# experiments/exp_e3_lobo_leakfree/status_e3.sh
#
# Read-only status check for the E3 leak-free LOBO protocol.
# Glob is intentionally restricted to results/e3_lobo_leakfree_2* so this
# script can NEVER touch/list any pre-existing results/exp_xjtu_lobo_* dirs
# produced by the original (non-leak-free) protocol.
#
# Usage:
#   bash experiments/exp_e3_lobo_leakfree/status_e3.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GLOB="$PROJECT_ROOT/results/e3_lobo_leakfree_2*"

shopt -s nullglob
RUN_DIRS=($GLOB)
shopt -u nullglob

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[status_e3] no results/e3_lobo_leakfree_2* directories found yet."
  exit 0
fi

for run_dir in "${RUN_DIRS[@]}"; do
  echo "============================================================"
  echo "[status_e3] run: $run_dir"
  for arm_dir in "$run_dir"/*/; do
    [[ -d "$arm_dir" ]] || continue
    arm="$(basename "$arm_dir")"
    n_fold_json=$(find "$arm_dir" -maxdepth 1 -name 'fold*_seed*.json' 2>/dev/null | wc -l)
    n_ckpt=$(find "$arm_dir/checkpoints" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)
    has_summary="no"
    [[ -f "$arm_dir/summary.json" ]] && has_summary="yes"
    echo "  arm=$arm  fold*_seed*.json=$n_fold_json  checkpoints=$n_ckpt  summary.json=$has_summary"
  done
done
