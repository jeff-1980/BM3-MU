#!/bin/bash
# experiments/exp_e1_dual_baselines/run_e1b_n8.sh
#
# E1b n=8 extension (third-round review, M1): XJTU-SY Cond2->Cond3
# cross-condition, six arms (single/dual x {cnn1d, bm2, bm3-CEonly}),
# add seeds 5/6/7 (existing seeds 0-4 untouched), 6 configs x 3 seeds
# = 18 runs total. Idempotent (skips a config if its summary.json exists).
#
# Pre-registration: prereg_e1b_n8.md (written before this script's first run).
#
# Usage:
#   tmux new -s e1b_n8
#   cd "$(dirname "$0")/../.." && source venv/bin/activate
#   bash experiments/exp_e1_dual_baselines/run_e1b_n8.sh 2>&1 | tee results/e1b_n8_run_$(date +%Y%m%d-%H%M%S).log

set -euo pipefail
cd "$(dirname "$0")/../.."
source venv/bin/activate

TRAIN="experiments/exp_xjtu/train.py"

CONFIGS=(
  "experiments/exp_mext_e13_1dcnn_xjtu_cross/config_newseed.yaml|results/exp_mext_e13_1dcnn_xjtu_cross_newseed"
  "experiments/exp_e1_dual_baselines/xjtu_cross_dual_cnn/config_cross_dual_cnn_newseed.yaml|results/exp_e1b_xjtu_cross_dual_cnn_newseed"
  "experiments/exp_e1_dual_baselines/xjtu_cross_single_bm2/config_cross_single_bm2_newseed.yaml|results/exp_e1b_xjtu_cross_single_bm2_newseed"
  "experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/config_cross_dual_bm2_newseed.yaml|results/exp_e1_xjtu_cross_dual_bm2_newseed"
  "experiments/exp_xjtu/config_cross_nokin_newseed.yaml|results/exp_xjtu_cross_nokin_newseed"
  "experiments/exp_xjtu/config_cross_dual_nokin_newseed.yaml|results/exp_xjtu_cross_dual_nokin_newseed"
)

echo "=== E1b n=8 extension: 6 configs, 18 runs total ==="
for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%|*}"
  rdir="${entry##*|}"
  echo
  echo "============================================================"
  echo "[E1b-n8] config=$cfg  results_dir=$rdir"
  echo "============================================================"
  if [ -f "$rdir/summary.json" ]; then
    echo "  [SKIP] $rdir/summary.json already exists"
    continue
  fi
  python3 "$TRAIN" --config "$cfg"
done

echo
echo "=== E1b n=8 queue done, running merge+Wilcoxon analysis ==="
python3 experiments/exp_e1_dual_baselines/analyze_e1b_n8.py

echo
echo "=== E1b n=8 EXTENSION COMPLETE ==="
