# BearMamba-3

Multi-sensor vibration fusion for bearing fault diagnosis on the official Mamba-3
state-space backbone (ICLR 2026), with an exploratory kinematic frequency-alignment
regulariser (L_kin) that exploits Mamba-3's complex oscillatory states. Code and
configurations accompanying the manuscript submitted to *IEEE Sensors Journal*.

## What's in this repository

```
bearmamba3/       model architecture, kinematic loss, noise generators, data pipelines (CWRU/PU/XJTU-SY)
baselines/        comparison backbones: 1D-CNN (+ LayerNorm / no-BN / attention-fusion variants),
                  Transformer-1D, BearMamba-2, spectral-kurtosis SVM
experiments/      one subdirectory per experiment: train.py / config_*.yaml / run_*.sh
paper_artifacts/  aggregated results (summary.json / CSV / figures) underlying every
                  table and figure in the paper — see "Paper artifacts" below
requirements.txt  core dependencies to install
requirements-freeze.txt   full pinned environment (177 packages) for exact reproduction
prereg_e1b_n8.md  pre-registration for the E1b n=8 stress-test extension (locked
                  before that run started; cited by name in the manuscript)
```

Raw training logs, per-seed checkpoints, and the full experimental result tree
(including large per-epoch `.npz` state-frequency snapshots, ~1.7 GB) are **not**
included here; `paper_artifacts/` has the aggregated numbers that produced every
reported figure, and everything is reproducible from the code + configs in this
repository given the public datasets below.

## Environment

Tested on WSL2 Ubuntu, Python 3.12, CUDA 12.1, RTX A5000 (Ampere, sm_86).

```bash
python3 -m venv venv && source venv/bin/activate

# PyTorch first, matched to your CUDA version:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121

# mamba_ssm must be built from source — the PyPI wheel does not include
# the Mamba-3 module used by this project:
MAMBA_FORCE_BUILD=TRUE pip install "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"

pip install -r requirements.txt
```

For an exact, fully-pinned environment instead, use `pip install -r requirements-freeze.txt`
after installing torch/mamba_ssm as above.

## Datasets

This repository does not redistribute any dataset. Download each from its official
source and place it as shown (paths match what the configs in `experiments/` expect):

| Dataset | Official source | Expected path |
|---|---|---|
| CWRU (Case Western Reserve University) | <https://engineering.case.edu/bearingdatacenter> | `<repo_root>/data/cwru_12k_de/` |
| Paderborn University (PU) | <https://mb.uni-paderborn.de/kat/forschung/kat-datacenter> | `~/data_pu/` |
| XJTU-SY | IEEE DataPort, *XJTU-SY Rolling Element Bearing Accelerated Life Test Datasets* | `~/data_xjtu/XJTU-SY_Bearing_Datasets/` |

Note the PU and XJTU-SY paths are outside the repository (in the user's home
directory) — this matches the path convention baked into `experiments/exp06_pu/`
and `experiments/exp_xjtu/`'s configs; only CWRU is expected inside the repo tree.
CWRU needs to be pre-processed into per-class 12kHz DE(+FE) `.mat`-derived windows
under `cwru_12k_de/` (see `bearmamba3/data_cwru.py` for the exact file/label
convention expected).

## Reproducing a result

All commands are run from the repository root, with the venv active:

```bash
# Single-sensor CWRU baseline (Table I / tab:cwru_baselines, BM3 CE-only row)
python experiments/exp01_cwru_baseline/train.py --config experiments/exp01_cwru_baseline/config.yaml

# Dual-sensor (DE+FE) CWRU, CE-only, at 0 dB SNR (Table tab:b2)
python experiments/exp01_cwru_baseline/train.py --config experiments/exp_b2_dual_sensor/config_dual_nokin.yaml

# Full B2 SNR sweep (both arms, both L_kin settings, 5 seeds each), idempotent:
python experiments/exp_b2_dual_sensor/run_b2_snr_curve.py

# XJTU-SY leave-one-bearing-out, leakage-free protocol (Table tab:xjtu / tab:xjtu_lobo_strat):
bash experiments/exp_e3_lobo_leakfree/run_lobo_leakfree.sh
```

Every `experiments/<exp_name>/` directory is self-contained: its `config*.yaml`
files declare `results_dir`, seeds, SNR, and hyperparameters; most experiments
reuse a shared `train.py` (`exp01_cwru_baseline/train.py` for CWRU, `exp06_pu/train.py`
for Paderborn, `exp_xjtu/train.py` for XJTU-SY) via `--config`, invoked either
directly or through that experiment's `run_*.py`/`run_*.sh` orchestrator, which
also handles idempotent skip-if-done and (where relevant) the analysis/plotting
pass. `--smoke` runs a 2-epoch, single-seed pass for a fast sanity check before
committing to a full run.

## Paper artifacts

`paper_artifacts/results/` mirrors the original `results/<experiment>/` layout,
but keeps only the **aggregated** files that were read to produce the paper's
numbers — `summary.json` (mean/std/per-seed values), small CSV/Markdown reports,
and every figure image (`paper_artifacts/results/figures/`). It deliberately
excludes per-seed raw training logs and the per-epoch `.npz` state-frequency
snapshots (~1.7 GB across the full experimental tree) — those are regenerable
by rerunning the corresponding `experiments/` script against the public
datasets above, but are too large to distribute here.

The six `*_newseed` directories (`exp_mext_e13_1dcnn_xjtu_cross_newseed`,
`exp_e1b_xjtu_cross_dual_cnn_newseed`, `exp_e1b_xjtu_cross_single_bm2_newseed`,
`exp_e1_xjtu_cross_dual_bm2_newseed`, `exp_xjtu_cross_nokin_newseed`,
`exp_xjtu_cross_dual_nokin_newseed`) are the seeds-5/6/7 stress-test extension
for the E1b XJTU-SY cross-condition experiment (pre-registered in
`prereg_e1b_n8.md`); merged with the original seeds 0-4 they are the direct
source of the manuscript's n=8 numbers in Table `tab:e1b_backbone_agnostic`
(the three-backbone fusion-gain significance test) and the regime-selection
table (dual-BM2 vs. dual-BM3 on XJTU-SY).

Pointers from paper section to artifact (non-exhaustive; every number in the
paper traces back to a `summary.json` in this tree):

| Paper item | Source |
|---|---|
| Table I (CWRU baselines) | `paper_artifacts/results/{exp07_*,exp01_cwru_baseline,exp01_cwru_kin}/summary.json` |
| CWRU SNR ablation (BM2/BM3±L_kin) | `paper_artifacts/results/exp02_snr*_{nokin,kin}/summary.json`, `exp04_mamba2_snr*/summary.json` |
| PU cross-condition | `paper_artifacts/results/exp06_pu_{nokin,kin,bm2_nokin}/summary.json` |
| CWRU dual-sensor SNR curve (B2, incl. n=8 extension) | `paper_artifacts/results/exp_b2_dual_*/summary.json`, `exp_e6_*_newseed/summary.json` |
| CWRU pink-noise check (E5) | `paper_artifacts/results/exp_e5_{single,dual}_pink_snr*/summary.json` |
| XJTU-SY LOBO / cross-condition (leakage-free) | `paper_artifacts/results/e3_lobo_leakfree_20260709-205255/*/summary.json` |
| XJTU-SY LOBO / cross-condition (original protocol, superseded) | `paper_artifacts/results/exp_xjtu_{lobo,cross}*_{nokin,kin}/summary.json` |
| Three-backbone OOD fusion gain (E1b, original $n=5$) | `paper_artifacts/results/e1_dual_baselines_20260709-1052/`, `e1b_fillcells_20260709-1515/`, `e1_comparison_20260709-1431/e1_comparison.md` |
| Three-backbone OOD fusion gain (E1b, $n=8$ stress-test extension — current headline numbers in the paper) | `prereg_e1b_n8.md` (pre-registration); `paper_artifacts/results/{exp_mext_e13_1dcnn_xjtu_cross,exp_e1b_xjtu_cross_dual_cnn,exp_e1b_xjtu_cross_single_bm2,exp_e1_xjtu_cross_dual_bm2,exp_xjtu_cross_nokin,exp_xjtu_cross_dual_nokin}_newseed/summary.json` (seeds 5-7, merge with the $n=5$ row above by seed id); `experiments/exp_e1_dual_baselines/run_e1b_n8.sh` + `analyze_e1b_n8.py` reproduce the run and the merge/Wilcoxon analysis end to end |
| BatchNorm/LayerNorm ablations (Phase 2/2b, CNN-LN) | `paper_artifacts/results/exp_mext_e2{1,1b,2,3}_*/summary.json`, `e2_cnn_ln_20260709-1723/` |
| 1D-CNN OOD stress test (Phase 1) | `paper_artifacts/results/exp_mext_e1{1,2,3,4}_*/summary.json` |
| Channel-coherence mechanism analysis (A1) | `paper_artifacts/results/a1_coherence_20260708-2358/` (`a1_summary.md`, per-cell CSVs, `run_a1_coherence.py`) |
| Forward-only throughput microbenchmark (S6) | `paper_artifacts/results/step5_throughput_bench_20260721/results.json` |

## Citation

```bibtex
@article{TODO_bearmamba3_sensors,
  title   = {BearMamba-3: Multi-Sensor Vibration Fusion with Mamba-3
             State-Space Models for Robust Bearing Fault Diagnosis},
  author  = {TODO},
  journal = {IEEE Sensors Journal},
  year    = {TODO},
  note    = {under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
