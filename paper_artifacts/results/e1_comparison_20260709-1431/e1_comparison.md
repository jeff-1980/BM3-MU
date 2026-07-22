# E1 comparison: dual-sensor baseline arms

Output dir: `results/e1_comparison_20260709-1431`. Read-only assembly from existing `results/*/summary.json`; no training run, no existing file modified.

## Table 1 — CWRU, per SNR level (mean ± std %, 5 seeds, best_val_acc)

| arm | SNR=0dB | SNR=-2dB | SNR=-4dB | SNR=-6dB | SNR=-8dB |
|---|---|---|---|---|---|
| dual-BM3 (CE-only) | 99.88 ± 0.09 | 99.37 ± 0.27 | 98.06 ± 0.48 | 94.24 ± 0.79 | 87.99 ± 1.97 |
| dual-BM3 (+Lkin) | 99.93 ± 0.06 | 99.35 ± 0.28 | 98.35 ± 0.65 | 94.48 ± 0.76 | 87.26 ± 1.17 |
| dual-BM2 | 99.98 ± 0.03 | 99.88 ± 0.12 | 99.61 ± 0.16 | 98.45 ± 0.28 | 95.07 ± 0.50 |
| dual-CNN (e14) | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.98 ± 0.03 | 98.91 ± 0.18 |
| cnn1d_attnfusion | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.98 ± 0.03 | 99.90 ± 0.10 | 98.90 ± 0.23 |
| single-BM3 (CE-only) | 99.75 ± 0.19 | 99.32 ± 0.31 | 97.37 ± 0.90 | 92.49 ± 1.75 | 85.45 ± 2.60 |
| single-BM3 (+Lkin) | 99.81 ± 0.11 | 99.34 ± 0.21 | 97.57 ± 0.34 | 92.40 ± 1.51 | 85.76 ± 1.82 |
| single-BM2 | 99.86 ± 0.10 | 99.51 ± 0.48 | 98.81 ± 0.48 | 95.79 ± 1.23 | 91.03 ± 0.84 |
| single-CNN | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.15 ± 0.00 | 96.01 ± 0.00 |

## Table 2 — XJTU Cond2->Cond3 cross-condition (mean ± std %, 5 seeds)

| arm | macro_f1 | macro_recall |
|---|---|---|
| dual-BM3 (CE-only) | 94.90 ± 4.52 | 93.80 ± 5.58 |
| dual-BM3 (+Lkin) | 94.78 ± 4.22 | 93.73 ± 5.32 |
| dual-BM2 | 83.54 ± 7.46 | 81.10 ± 7.72 |
| single-BM3 (CE-only) | 80.89 ± 5.21 | 78.17 ± 5.04 |
| single-BM3 (+Lkin) | 79.73 ± 7.39 | 77.37 ± 7.23 |
| single-CNN | 41.31 ± 0.57 | 50.59 ± 0.27 |
| single-BM2 | n/a | n/a |
| dual-CNN | n/a | n/a |

`single-BM2` and `dual-CNN` rows for XJTU are `n/a`: confirmed no such config exists anywhere under `experiments/exp_xjtu/` or `experiments/exp_e1_dual_baselines/` (grepped all their `*.yaml` for `backbone`). Not fabricated, not estimated.

## Task 3a — dual-BM2 vs dual-BM3(CE-only), per-seed (report-only Wilcoxon)

### dual-BM2 vs dual-BM3(CE-only) @ CWRU@-6dB

- concordant: 5/5 (A>B)
- Wilcoxon signed-rank: stat=0.0, p=0.0625
- per-seed CSV: `pair_dualBM2_vs_dualBM3_CWRU_m6dB.csv`

| seed | dual-BM2 | dual-BM3(CE-only) | diff_a_minus_b |
|---|---|---|---|
| 0 | 0.9881 | 0.9473 | 0.0408 |
| 1 | 0.9856 | 0.9516 | 0.0340 |
| 2 | 0.9805 | 0.9448 | 0.0357 |
| 3 | 0.9864 | 0.9286 | 0.0578 |
| 4 | 0.9822 | 0.9397 | 0.0425 |

### dual-BM2 vs dual-BM3(CE-only) @ CWRU@-8dB

- concordant: 5/5 (A>B)
- Wilcoxon signed-rank: stat=0.0, p=0.0625
- per-seed CSV: `pair_dualBM2_vs_dualBM3_CWRU_m8dB.csv`

| seed | dual-BM2 | dual-BM3(CE-only) | diff_a_minus_b |
|---|---|---|---|
| 0 | 0.9592 | 0.9082 | 0.0510 |
| 1 | 0.9465 | 0.8912 | 0.0552 |
| 2 | 0.9456 | 0.8828 | 0.0629 |
| 3 | 0.9490 | 0.8522 | 0.0969 |
| 4 | 0.9533 | 0.8649 | 0.0884 |

### dual-BM2 vs dual-BM3(CE-only) @ XJTU Cond2->Cond3 (macro_f1)

- concordant: 5/5 (B>A)
- Wilcoxon signed-rank: stat=0.0, p=0.0625
- per-seed CSV: `pair_dualBM2_vs_dualBM3_XJTUcross.csv`

| seed | dual-BM2 | dual-BM3(CE-only) | diff_a_minus_b |
|---|---|---|---|
| 0 | 0.7476 | 0.8668 | -0.1193 |
| 1 | 0.9274 | 0.9929 | -0.0655 |
| 2 | 0.7648 | 0.9383 | -0.1736 |
| 3 | 0.8216 | 0.9837 | -0.1621 |
| 4 | 0.9158 | 0.9630 | -0.0472 |

## Task 3b — cnn1d_attnfusion vs dual-CNN, per SNR (report-only Wilcoxon)

Uses the e1-batch dual-CNN re-run (`cwru_dual_cnn_snr*`, trained in the same session as attnfusion) rather than the original `exp_mext_e14_1dcnn_cwru_dual_snr*` copy, for a same-session apples-to-apples pairing. See `dualCNN_rerun_vs_original_e14.csv` for how much these two independent runs of the identical config/seeds differ from each other (repeat-run noise floor).

### cnn1d_attnfusion vs dual-CNN(e1 rerun) @ SNR=0dB

- concordant: 0/5 (A>B)
- Wilcoxon signed-rank: stat=nan, p=nan
- per-seed CSV: `pair_attnfusion_vs_dualCNN_snr0.csv`

| seed | cnn1d_attnfusion | dual-CNN(e1 rerun) | diff_a_minus_b |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 1.0000 | 0.0000 |
| 2 | 1.0000 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 1.0000 | 0.0000 |
| 4 | 1.0000 | 1.0000 | 0.0000 |

### cnn1d_attnfusion vs dual-CNN(e1 rerun) @ SNR=-2dB

- concordant: 0/5 (A>B)
- Wilcoxon signed-rank: stat=nan, p=nan
- per-seed CSV: `pair_attnfusion_vs_dualCNN_snrm2.csv`

| seed | cnn1d_attnfusion | dual-CNN(e1 rerun) | diff_a_minus_b |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 1.0000 | 0.0000 |
| 2 | 1.0000 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 1.0000 | 0.0000 |
| 4 | 1.0000 | 1.0000 | 0.0000 |

### cnn1d_attnfusion vs dual-CNN(e1 rerun) @ SNR=-4dB

- concordant: 1/5 (B>A)
- Wilcoxon signed-rank: stat=0.0, p=1.0
- per-seed CSV: `pair_attnfusion_vs_dualCNN_snrm4.csv`

| seed | cnn1d_attnfusion | dual-CNN(e1 rerun) | diff_a_minus_b |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 1.0000 | 0.0000 |
| 2 | 1.0000 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 1.0000 | 0.0000 |
| 4 | 0.9992 | 1.0000 | -0.0008 |

### cnn1d_attnfusion vs dual-CNN(e1 rerun) @ SNR=-6dB

- concordant: 3/5 (B>A)
- Wilcoxon signed-rank: stat=0.0, p=0.25
- per-seed CSV: `pair_attnfusion_vs_dualCNN_snrm6.csv`

| seed | cnn1d_attnfusion | dual-CNN(e1 rerun) | diff_a_minus_b |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.0000 |
| 1 | 0.9992 | 1.0000 | -0.0008 |
| 2 | 0.9983 | 1.0000 | -0.0017 |
| 3 | 0.9975 | 0.9992 | -0.0017 |
| 4 | 1.0000 | 1.0000 | 0.0000 |

### cnn1d_attnfusion vs dual-CNN(e1 rerun) @ SNR=-8dB

- concordant: 2/5 (B>A)
- Wilcoxon signed-rank: stat=1.0, p=0.5
- per-seed CSV: `pair_attnfusion_vs_dualCNN_snrm8.csv`

| seed | cnn1d_attnfusion | dual-CNN(e1 rerun) | diff_a_minus_b |
|---|---|---|---|
| 0 | 0.9873 | 0.9873 | 0.0000 |
| 1 | 0.9898 | 0.9898 | 0.0000 |
| 2 | 0.9932 | 0.9915 | 0.0017 |
| 3 | 0.9873 | 0.9907 | -0.0034 |
| 4 | 0.9873 | 0.9898 | -0.0025 |

## Repeat-run noise floor: e1-batch dual-CNN re-run vs original exp_mext_e14

Same config, same seeds (0-4), two independent training invocations (`run_dual_baselines.sh`'s idempotency check only looks at the new `e1_dual_baselines_<ts>/` path, so this arm was NOT skipped and was retrained from scratch rather than reusing `exp_mext_e14`'s numbers).

| snr | seed | e1_rerun | original_e14 | diff |
|---|---|---|---|---|
| 0 | 0 | 1.0000 | 1.0000 | 0.0000 |
| 0 | 1 | 1.0000 | 1.0000 | 0.0000 |
| 0 | 2 | 1.0000 | 1.0000 | 0.0000 |
| 0 | 3 | 1.0000 | 1.0000 | 0.0000 |
| 0 | 4 | 1.0000 | 1.0000 | 0.0000 |
| -2 | 0 | 1.0000 | 1.0000 | 0.0000 |
| -2 | 1 | 1.0000 | 1.0000 | 0.0000 |
| -2 | 2 | 1.0000 | 1.0000 | 0.0000 |
| -2 | 3 | 1.0000 | 1.0000 | 0.0000 |
| -2 | 4 | 1.0000 | 1.0000 | 0.0000 |
| -4 | 0 | 1.0000 | 1.0000 | 0.0000 |
| -4 | 1 | 1.0000 | 1.0000 | 0.0000 |
| -4 | 2 | 1.0000 | 1.0000 | 0.0000 |
| -4 | 3 | 1.0000 | 1.0000 | 0.0000 |
| -4 | 4 | 1.0000 | 1.0000 | 0.0000 |
| -6 | 0 | 1.0000 | 1.0000 | 0.0000 |
| -6 | 1 | 1.0000 | 1.0000 | 0.0000 |
| -6 | 2 | 1.0000 | 1.0000 | 0.0000 |
| -6 | 3 | 0.9992 | 0.9992 | 0.0000 |
| -6 | 4 | 1.0000 | 1.0000 | 0.0000 |
| -8 | 0 | 0.9873 | 0.9856 | 0.0017 |
| -8 | 1 | 0.9898 | 0.9907 | -0.0008 |
| -8 | 2 | 0.9915 | 0.9898 | 0.0017 |
| -8 | 3 | 0.9907 | 0.9898 | 0.0008 |
| -8 | 4 | 0.9898 | 0.9898 | 0.0000 |

