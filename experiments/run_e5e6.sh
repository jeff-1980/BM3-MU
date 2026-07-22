#!/bin/bash
# experiments/run_e5e6.sh
#
# E5 (pink-noise generalisation) + E6 (flagship-cell new-seed extension),
# run serially in one tmux session. 18 train.py invocations, 66 total runs:
#
#   E5 — CWRU dual vs single (CE-only), pink noise, SNR {-4,-6,-8} x 5 seeds
#        = 2 arms x 3 SNR = 6 configs x 5 seeds = 30 runs
#        configs: experiments/exp_e5_pink_noise/{single,dual}/config_snr{-4,-6,-8}.yaml
#        noise injection: bearmamba3/noise.py::add_noise_at_snr(noise_type="pink"),
#        spectral verification: experiments/exp_e5_pink_noise/verify_pink_spectrum.py
#        (already run once, PASS: pink fit slope -0.998 vs ideal -1.00; plot at
#        results/figures/e5_pink_noise_spectrum_verification.{png,pdf})
#
#   E6 — CWRU {single,dual} x {kin,nokin}, AWGN, SNR {-4,-6,-8} x NEW seeds {5,6,7}
#        = 4 arm-types x 3 SNR = 12 configs x 3 seeds = 36 runs
#        configs: experiments/exp_e6_newseeds/{single_nokin,single_kin,dual_nokin,dual_kin}/config_snr{-4,-6,-8}.yaml
#        field-identical to the existing flagship configs (exp02_snr*_{nokin,kin}.yaml,
#        exp_b2_dual_{nokin,kin}[_snrm*]) except seeds: [5,6,7] and a new results_dir
#        (existing seed 0-4 results are NOT touched/rerun; this only adds seeds 5-7).
#
# Usage:
#   tmux new -s e5e6
#   cd "$(dirname "$0")/.." && source venv/bin/activate
#   bash experiments/run_e5e6.sh 2>&1 | tee results/e5e6_run_$(date +%Y%m%d-%H%M%S).log
#   # then (same session, queued to auto-fire after this finishes):
#   bash experiments/report_e5e6_to_vault.sh 2>&1 | tee results/e5e6_report_$(date +%Y%m%d-%H%M%S).log

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

TRAIN=experiments/exp01_cwru_baseline/train.py

echo "=== E5+E6 queue: 18 configs, 66 runs total ==="

echo ""
echo "############################################################"
echo "# E5: pink noise, dual vs single CE-only, SNR {-4,-6,-8} x 5 seeds"
echo "############################################################"
for arm in single dual; do
  for snr in -4 -6 -8; do
    cfg="experiments/exp_e5_pink_noise/${arm}/config_snr${snr}.yaml"
    echo ""
    echo "============================================================"
    echo "[E5] arm=${arm} snr=${snr}  cfg=${cfg}"
    echo "============================================================"
    python "$TRAIN" --config "$cfg"
  done
done

echo ""
echo "############################################################"
echo "# E6: flagship cells, new seeds {5,6,7}, SNR {-4,-6,-8}"
echo "############################################################"
for arm in single_nokin single_kin dual_nokin dual_kin; do
  for snr in -4 -6 -8; do
    cfg="experiments/exp_e6_newseeds/${arm}/config_snr${snr}.yaml"
    echo ""
    echo "============================================================"
    echo "[E6] arm=${arm} snr=${snr}  cfg=${cfg}"
    echo "============================================================"
    python "$TRAIN" --config "$cfg"
  done
done

echo ""
echo "=== E5+E6 queue DONE ==="
