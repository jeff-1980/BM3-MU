#!/bin/bash
# experiments/report_e5e6_to_vault.sh
#
# Meant to be queued in the SAME tmux session right after run_e5e6.sh (type-
# ahead into the terminal — it sits in the tty input buffer and only actually
# executes once run_e5e6.sh's foreground process exits, same pattern used for
# E3's report_to_vault.sh). Appends (does not overwrite) the E5+E6 results
# section to the vault page.

set -u
cd "$(dirname "$0")/.."

VAULT_MD="/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/管家/回传-加固实验.md"

echo "$(date '+%F %T')  report_e5e6_to_vault: assembling E5+E6 section"
python experiments/build_e5e6_report.py --append-to "$VAULT_MD"
echo "$(date '+%F %T')  report_e5e6_to_vault: done, appended to ${VAULT_MD}"
