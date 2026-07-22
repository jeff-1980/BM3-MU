#!/bin/bash
# experiments/exp_e3_lobo_leakfree/report_to_vault.sh
#
# Intended to run AFTER run_lobo_leakfree.sh in the same tmux session (e3fix):
# polls every 30 minutes for all 80 leakfree runs (4 arms x 4 folds x 5 seeds,
# single_nokin already complete) to land under LEAKFREE_DIR. While waiting,
# overwrites the vault page with just a status header (current N/80 + ETA).
# Once 80/80, calls build_p3_report.py (read-only) to assemble the full P3
# comparison against the original protocol and overwrite the vault page with
# the complete report, stamped with a completion timestamp.
#
# Usage (from <repo_root>, venv active):
#   bash experiments/exp_e3_lobo_leakfree/report_to_vault.sh <leakfree_dir>

set -u
cd "$(dirname "$0")/../.."

LEAKFREE_DIR="${1:?usage: report_to_vault.sh <leakfree_dir>}"
VAULT_MD="/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/管家/回传-加固实验.md"
BUILD_PY="experiments/exp_e3_lobo_leakfree/build_p3_report.py"

echo "$(date '+%F %T')  report_to_vault: watching ${LEAKFREE_DIR} for 80/80, polling every 30min"

while true; do
  python "$BUILD_PY" --leakfree-dir "$LEAKFREE_DIR" --out "$VAULT_MD" --mode status
  N_DONE=$(find "$LEAKFREE_DIR" -name "fold*_seed*.json" 2>/dev/null | wc -l)
  echo "$(date '+%F %T')  status written: ${N_DONE}/80"
  if [ "$N_DONE" -ge 80 ]; then
    break
  fi
  sleep 1800
done

echo "$(date '+%F %T')  80/80 complete — assembling final P3 report"
python "$BUILD_PY" --leakfree-dir "$LEAKFREE_DIR" --out "$VAULT_MD" --mode report \
  --completed-ts "$(date '+%Y-%m-%d %H:%M:%S')"
echo "$(date '+%F %T')  report_to_vault: done, wrote ${VAULT_MD}"
