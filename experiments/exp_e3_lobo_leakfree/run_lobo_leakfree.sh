#!/usr/bin/env bash
# experiments/exp_e3_lobo_leakfree/run_lobo_leakfree.sh
#
# Launches the E3 leak-free LOBO matrix (4 folds x 4 arms x 5 seeds = 80 runs,
# estimated ~5.9-6.2 GPU-hours, see e3_cost_estimate.md).
#
# Usage:
#   source venv/bin/activate
#   bash experiments/exp_e3_lobo_leakfree/run_lobo_leakfree.sh [--dry-run]
#
# Each arm resumes automatically: train_lobo_leakfree.py skips any
# fold{N}_seed{S}.json that already exists in its results_dir, so this script
# is safe to re-invoke after a partial run / interruption.
#
# results_dir is parameterized via --results-dir (train_lobo_leakfree.py flag,
# added 2026-07-09) instead of relying on each config_*.yaml's own hardcoded
# path. Previously the 4 configs hardcoded results_dir to a fixed
# results/e3_lobo_leakfree_<one-time-smoke-timestamp>/<arm> path; a later
# invocation of this script collided with a chmod-444 summary.json left over
# from the original smoke run (single_nokin arm), crashing after arm 1 with
# zero effect on the remaining 3 arms. TS below defaults to a fresh timestamp
# per invocation so this can't recur; export TS=<existing-dir-timestamp>
# before calling this script to target/resume a specific prior batch dir
# instead of minting a new one (e.g. to continue results/e3_lobo_leakfree_<ts>/
# where an arm's own data already lives, as with single_nokin here).
#
# single_nokin is EXCLUDED from CONFIGS below: its full 4-fold x 5-seed matrix
# was already completed under results/e3_lobo_leakfree_20260709-205255/single_nokin/
# (20/20 runs, copied+reconciled from the original smoke-collision batch,
# summary.json regenerated via this same script's own aggregation code path
# skip-logic — see README_e3_single_nokin_recovery.md if present). Re-running
# it here would be redundant GPU time, not a correctness issue (the skip
# logic would just no-op it), but it's cleaner to not even invoke it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)/results/e3_lobo_leakfree_${TS}"
echo "[run_lobo_leakfree] TS=${TS}  OUT_ROOT=${OUT_ROOT}"

# config filename -> arm subdirectory name (matches each config's own
# canonical results_dir convention: single_nokin/single_kin/dual_nokin/dual_kin)
declare -A ARM_DIR=(
  [config_lobo_kin_leakfree.yaml]="single_kin"
  [config_lobo_dual_nokin_leakfree.yaml]="dual_nokin"
  [config_lobo_dual_kin_leakfree.yaml]="dual_kin"
)

CONFIGS=(
  config_lobo_kin_leakfree.yaml
  config_lobo_dual_nokin_leakfree.yaml
  config_lobo_dual_kin_leakfree.yaml
)

echo "[run_lobo_leakfree] configs to run (in order): ${CONFIGS[*]}"
echo "[run_lobo_leakfree] (single_nokin excluded — already complete, see header comment)"
echo "[run_lobo_leakfree] estimated total cost: see e3_cost_estimate.md (~5.9-6.2 GPU-hours; this invocation covers 3/4 arms, ~4.5h)"

for cfg in "${CONFIGS[@]}"; do
  arm_dir="${OUT_ROOT}/${ARM_DIR[$cfg]}"
  echo ""
  echo "============================================================"
  echo "[run_lobo_leakfree] arm: $cfg  ->  ${arm_dir}"
  echo "============================================================"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would run: python train_lobo_leakfree.py --config $cfg --results-dir ${arm_dir}"
    continue
  fi
  python train_lobo_leakfree.py --config "$cfg" --results-dir "${arm_dir}"
done

echo ""
echo "[run_lobo_leakfree] done. Check results with: bash status_e3.sh"
