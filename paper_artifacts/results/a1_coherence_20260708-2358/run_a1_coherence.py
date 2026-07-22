#!/usr/bin/env python3
"""
results/a1_coherence_20260708-2358/run_a1_coherence.py

A1 channel-coherence analysis (read-only, zero training).

Reuses:
  - bearmamba3.data_cwru  : MANIFEST, DE_BEARING geometry, load_signal() (raw .mat reader)
  - bearmamba3.data_xjtu  : BEARING_CONDITION/BEARING_FAILURE/CONDITION_RPM/LABEL_MAP,
                            compute_fault_onset(), valid_bearings() (raw .csv reader)
  - bearmamba3.kinematic_loss.compute_fault_freqs()-equivalent formulas (BPFO/BPFI/BSF)

Design notes (read before interpreting outputs):

1. Spectral estimation uses the FULL raw per-file/per-bearing waveform (not the
   2048-sample classifier training window). A 2048-sample window gives exactly
   one Welch segment -- too noisy for a coherence/PSD estimate. CWRU files are
   ~120k samples (>50 segments at nperseg=2048); XJTU fault-phase CSVs are
   32768 samples each and are concatenated across all fault-phase files for a
   bearing. This is an analysis-appropriate deviation from the training
   pipeline's window length, not a re-implementation of it; channel selection,
   raw file paths, and bearing/condition/rpm bookkeeping are reused as-is.

2. CWRU fault-frequency bands (BPFO/BPFI/BSF, harmonics 1-3, +-5 Hz) always use
   DE_BEARING geometry (n_balls=9, d=7.938mm, D=39.040mm), never FE_BEARING --
   this matches the project's own established convention (see
   experiments/exp_b2_dual_sensor/config_dual_kin.yaml: "the fault is at the
   DE bearing", bearing_kwargs = DE params). The seeded fault is physically at
   the drive end regardless of which channel(s) are analysed.

3. CWRU cells = (fault_type in {normal,inner,ball,outer}) x (load_hp in 0..3),
   16 cells, pooling the 3 fault sizes (7/14/21 mil) within a (type,load) cell
   (task said "each fault class x each load"; sub-dividing further by size
   would give 40 near-singleton cells with too little data for Welch/xcorr).
   "normal" has no seeded fault but the same geometry-derived bands are still
   evaluated as a built-in negative control (expect near-flat SNR there).

4. XJTU cells = (condition in {35Hz12kN, 37.5Hz11kN, 40Hz10kN}) x
   (fault_type in {OR,IR}), using only the official OR/IR bearings (Cage/Mixed
   excluded, per data_xjtu.py), fault-phase windows only (kurtosis onset,
   reused from compute_fault_onset). Geometry: LDK UER204, n_balls=8,
   d=7.92mm, D=34.55mm, alpha=0 (from data_xjtu.py's own module docstring).

5. Gain-vs-coherence correlation (task 3) needs *matched* (coherence, gain)
   pairs. The existing results/*/summary.json files do NOT break dual-vs-
   single gain down by fault-type/load/bearing -- they are aggregate 4-class
   (CWRU) or macro-F1/recall (XJTU) numbers per experiment directory. The
   finest granularity that actually exists in prior results is:
     - CWRU : dual-vs-single gain at each SNR level that was actually run for
       BOTH single (exp02_snr*_nokin) and dual (exp_b2_dual_nokin[_snrm*])
       CE-only pipelines: SNR in {-8,-6,-4,-2,0} dB (5 points).
       To pair this with a *matching* coherence measurement (coherence is a
       property of the raw signal, not of a training run), synthetic AWGN is
       added to the pooled clean DE/FE signal at the same SNR (reusing this
       project's own per-channel AWGN formula from data_cwru.py's
       CWRUDataset.__getitem__) purely to recompute band coherence -- no
       model is trained here, this is read-only spectral analysis on
       already-loaded arrays.
     - XJTU : only two real dual-vs-single comparisons exist at all --
       LOBO (Bearing3_{1,3,4,5}, Cond3 only) and Cond2->Cond3 cross-condition.
       These give exactly 2 (coherence, gain) points, paired against the
       Cond3-only mean coherence (LOBO) and the Cond2+Cond3 mean coherence
       (cross, since the fusion model sees both domains).
   This yields 5 CWRU points + 2 XJTU points = 7 total. N=7 (and N=2 for the
   XJTU-only subset) is very small; Spearman r on this is reported honestly
   as directional evidence only, not a confirmed relationship. This limitation
   is stated again in a1_summary.md rather than glossed over.

Outputs (this directory only; nothing outside results/a1_coherence_<ts>/ is
written, no existing file is modified):
  coherence_by_condition.csv
  per_channel_snr.csv
  gain_vs_coherence.{pdf,png}
  gain_vs_coherence_points.csv   (the raw (coherence, gain) pairs behind the plot)
  a1_summary.md
"""
import sys
import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parents[3]  # repo root when run from a full checkout with data/ in place
sys.path.insert(0, str(PROJ_ROOT))

from bearmamba3.data_cwru import MANIFEST, DE_BEARING, load_signal as cwru_load_signal, FS as CWRU_FS
from bearmamba3.data_xjtu import (
    BEARING_CONDITION, BEARING_FAILURE, CONDITION_RPM, LABEL_MAP,
    compute_fault_onset, valid_bearings, FS as XJTU_FS,
)

OUT_DIR = Path(__file__).parent
CWRU_DATA_DIR = PROJ_ROOT / "data" / "cwru_12k_de"
XJTU_DATA_ROOT = os.path.expanduser("~/data_xjtu/XJTU-SY_Bearing_Datasets")

# XJTU bearing geometry (LDK UER204, per bearmamba3/data_xjtu.py module docstring)
XJTU_BEARING = dict(n_balls=8, d=7.92, D=34.55, contact_angle_deg=0.0)

N_HARMONICS = 3
BAND_HALFWIDTH_HZ = 5.0
NPERSEG = 2048


# ───────────────────────────── fault-frequency helpers ──────────────────────

def fault_freqs(rpm, n_balls, d, D, contact_angle_deg=0.0, n_harmonics=N_HARMONICS):
    """Return dict {'BPFO': [h1,h2,h3], 'BPFI': [...], 'BSF': [...]} in Hz."""
    fr = rpm / 60.0
    ratio = d / D
    cos_a = np.cos(np.deg2rad(contact_angle_deg))
    bpfo = (n_balls / 2) * fr * (1 - ratio * cos_a)
    bpfi = (n_balls / 2) * fr * (1 + ratio * cos_a)
    bsf = (D / (2 * d)) * fr * (1 - (ratio * cos_a) ** 2)
    harms = np.arange(1, n_harmonics + 1)
    return {
        "BPFO": (bpfo * harms).tolist(),
        "BPFI": (bpfi * harms).tolist(),
        "BSF": (bsf * harms).tolist(),
    }


def band_mask(freqs_array, center, halfwidth=BAND_HALFWIDTH_HZ):
    return (freqs_array >= center - halfwidth) & (freqs_array <= center + halfwidth)


def all_fault_bands_mask(freqs_array, freq_dict):
    m = np.zeros_like(freqs_array, dtype=bool)
    for _, harmlist in freq_dict.items():
        for f0 in harmlist:
            m |= band_mask(freqs_array, f0)
    return m


def neighbor_noise_mask(freqs_array, freq_dict, inner=BAND_HALFWIDTH_HZ, outer=15.0):
    """Flanking bands [f0+inner, f0+outer] and [f0-outer, f0-inner] for each target freq,
    excluding anything that falls inside ANY fault band (avoids self-contamination
    between nearby harmonics)."""
    fault_m = all_fault_bands_mask(freqs_array, freq_dict)
    m = np.zeros_like(freqs_array, dtype=bool)
    for _, harmlist in freq_dict.items():
        for f0 in harmlist:
            lo1, hi1 = f0 - outer, f0 - inner
            lo2, hi2 = f0 + inner, f0 + outer
            m |= (freqs_array >= lo1) & (freqs_array <= hi1)
            m |= (freqs_array >= lo2) & (freqs_array <= hi2)
    return m & (~fault_m)


# ───────────────────────────── metric computation ────────────────────────────

def compute_cell_metrics(ch_a, ch_b, fs, freq_dict, label):
    """ch_a, ch_b: 1D float arrays, same length, same fs. Returns dict of metrics."""
    n = min(len(ch_a), len(ch_b))
    ch_a, ch_b = ch_a[:n], ch_b[:n]

    f_coh, Cxy = scipy.signal.coherence(ch_a, ch_b, fs=fs, nperseg=min(NPERSEG, n))
    fault_m = all_fault_bands_mask(f_coh, freq_dict)
    mean_coh_fault = float(np.mean(Cxy[fault_m])) if fault_m.any() else float("nan")
    mean_coh_all = float(np.mean(Cxy))

    snrs = {}
    for name, ch in (("A", ch_a), ("B", ch_b)):
        f_psd, Pxx = scipy.signal.welch(ch, fs=fs, nperseg=min(NPERSEG, n))
        fault_m2 = all_fault_bands_mask(f_psd, freq_dict)
        noise_m2 = neighbor_noise_mask(f_psd, freq_dict)
        sig_power = float(np.mean(Pxx[fault_m2])) if fault_m2.any() else float("nan")
        noise_power = float(np.mean(Pxx[noise_m2])) if noise_m2.any() else float("nan")
        snr_db = 10 * np.log10(sig_power / noise_power) if noise_power and noise_power > 0 else float("nan")
        snrs[name] = dict(sig_power=sig_power, noise_power=noise_power, snr_db=snr_db)

    # np.correlate(mode="full") is O(n^2) direct convolution -- with multi-
    # million-sample concatenated signals this stalls for minutes per cell.
    # scipy.signal.correlate(method="fft") computes the identical result in
    # O(n log n).
    xc = scipy.signal.correlate(ch_a - ch_a.mean(), ch_b - ch_b.mean(),
                                 mode="full", method="fft")
    norm = (np.linalg.norm(ch_a - ch_a.mean()) * np.linalg.norm(ch_b - ch_b.mean()))
    xc_norm = xc / norm if norm > 0 else xc
    peak_idx = int(np.argmax(np.abs(xc_norm)))
    peak_lag = peak_idx - (n - 1)
    xcorr_peak = float(xc_norm[peak_idx])

    return dict(
        label=label, n_samples=n, fs=fs,
        mean_coherence_fault_bands=mean_coh_fault,
        mean_coherence_all=mean_coh_all,
        snr_A_db=snrs["A"]["snr_db"], snr_B_db=snrs["B"]["snr_db"],
        sig_power_A=snrs["A"]["sig_power"], noise_power_A=snrs["A"]["noise_power"],
        sig_power_B=snrs["B"]["sig_power"], noise_power_B=snrs["B"]["noise_power"],
        xcorr_peak=xcorr_peak, xcorr_peak_lag_samples=peak_lag,
    )


def add_awgn(x, snr_db, rng):
    p = np.mean(x ** 2)
    noise_p = p / (10 ** (snr_db / 10.0))
    return x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_p)


# ───────────────────────────── CWRU ──────────────────────────────────────────

def cwru_cells():
    """Group MANIFEST ids by (fault_type, load_hp); returns dict[(type,load)] -> list[fid]."""
    groups = {}
    for fid, (ftype, fsize, load, rpm, label) in MANIFEST.items():
        groups.setdefault((ftype, load), []).append(fid)
    return groups


def cwru_load_de_fe(fids):
    """Concatenate DE and FE across a list of file ids. Returns (de, fe, rpm_mean)."""
    des, fes, rpms = [], [], []
    for fid in fids:
        path = CWRU_DATA_DIR / f"{fid}.mat"
        sig, rpm = cwru_load_signal(path, channels=["DE", "FE"])
        des.append(sig[0]); fes.append(sig[1]); rpms.append(rpm)
    return np.concatenate(des), np.concatenate(fes), float(np.mean(rpms))


def run_cwru(rows_coh, rows_snr):
    groups = cwru_cells()
    cwru_cache = {}  # (type,load) -> (de, fe, rpm)
    for (ftype, load), fids in sorted(groups.items()):
        print(f"  [CWRU] {ftype} load={load} ({len(fids)} files)...", flush=True)
        de, fe, rpm = cwru_load_de_fe(fids)
        cwru_cache[(ftype, load)] = (de, fe, rpm)
        freqs = fault_freqs(rpm, **DE_BEARING)
        m = compute_cell_metrics(de, fe, CWRU_FS, freqs, label=f"CWRU_{ftype}_load{load}")
        m.update(dataset="CWRU", fault_type=ftype, load_hp=load, rpm=rpm, n_files=len(fids))
        rows_coh.append(m)
        rows_snr.append(dict(dataset="CWRU", cell=f"{ftype}_load{load}",
                              channel_A="DE", channel_B="FE",
                              snr_A_db=m["snr_A_db"], snr_B_db=m["snr_B_db"],
                              sig_power_A=m["sig_power_A"], noise_power_A=m["noise_power_A"],
                              sig_power_B=m["sig_power_B"], noise_power_B=m["noise_power_B"]))
    return cwru_cache


def run_cwru_snr_sweep(cwru_cache, snr_list):
    """Pool ALL cwru cells' DE/FE into one big array, add AWGN at each snr_list level,
    recompute mean fault-band coherence. Returns dict snr -> mean_coherence."""
    all_de = np.concatenate([v[0] for v in cwru_cache.values()])
    all_fe = np.concatenate([v[1] for v in cwru_cache.values()])
    mean_rpm = float(np.mean([v[2] for v in cwru_cache.values()]))
    freqs = fault_freqs(mean_rpm, **DE_BEARING)
    out = {}
    rng = np.random.default_rng(0)
    for snr in snr_list:
        de_n = add_awgn(all_de, snr, rng)
        fe_n = add_awgn(all_fe, snr, rng)
        f_coh, Cxy = scipy.signal.coherence(de_n, fe_n, fs=CWRU_FS, nperseg=NPERSEG)
        fm = all_fault_bands_mask(f_coh, freqs)
        out[snr] = float(np.mean(Cxy[fm])) if fm.any() else float("nan")
    return out


# ───────────────────────────── XJTU ──────────────────────────────────────────

def xjtu_load_hv(bearing_name):
    cond = BEARING_CONDITION[bearing_name]
    onset_idx, _ = compute_fault_onset(XJTU_DATA_ROOT, bearing_name)
    folder = os.path.join(XJTU_DATA_ROOT, cond, bearing_name)
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")),
                       key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    fault_csvs = all_csvs[onset_idx:]
    # Some bearings (e.g. Bearing3_1) have 1000s of fault-phase files (32768
    # rows each); reading every one with a per-file parser made this analysis
    # take hours for no statistical benefit -- Welch/coherence estimates don't
    # need every file, just "enough" segments. Evenly subsample to a fixed cap
    # (still >1M samples per bearing, several times more than any CWRU cell
    # uses) and read with pandas' C parser (np.loadtxt is the slow part, not
    # the subsampling) instead of np.loadtxt.
    MAX_FILES = 40
    if len(fault_csvs) > MAX_FILES:
        idx = np.linspace(0, len(fault_csvs) - 1, MAX_FILES).astype(int)
        fault_csvs = [fault_csvs[i] for i in idx]
    hs, vs = [], []
    for f in fault_csvs:
        raw = pd.read_csv(f, skiprows=1, header=None, usecols=[0, 1],
                           dtype=np.float32).values
        hs.append(raw[:, 0]); vs.append(raw[:, 1])
    return np.concatenate(hs), np.concatenate(vs)


def run_xjtu(rows_coh, rows_snr):
    per_bearing = {}
    for cond in ("35Hz12kN", "37.5Hz11kN", "40Hz10kN"):
        rpm = CONDITION_RPM[cond]
        freqs = fault_freqs(rpm, **XJTU_BEARING)
        for bname in valid_bearings(cond):
            print(f"  [XJTU] {cond} {bname} (onset scan + load)...", flush=True)
            h, v = xjtu_load_hv(bname)
            print(f"  [XJTU] {cond} {bname} loaded, computing metrics...", flush=True)
            m = compute_cell_metrics(h, v, XJTU_FS, freqs, label=f"XJTU_{bname}")
            ftype = BEARING_FAILURE[bname]
            m.update(dataset="XJTU", condition=cond, fault_type=ftype, bearing=bname, rpm=rpm)
            per_bearing[bname] = m

    # aggregate per (condition, fault_type)
    cells = {}
    for bname, m in per_bearing.items():
        key = (m["condition"], m["fault_type"])
        cells.setdefault(key, []).append(m)
    cell_aggs = {}
    for (cond, ftype), ms in sorted(cells.items()):
        agg = dict(
            dataset="XJTU", condition=cond, fault_type=ftype,
            n_bearings=len(ms), bearings=",".join(m["bearing"] for m in ms),
            rpm=ms[0]["rpm"],
            mean_coherence_fault_bands=float(np.mean([m["mean_coherence_fault_bands"] for m in ms])),
            mean_coherence_all=float(np.mean([m["mean_coherence_all"] for m in ms])),
            snr_A_db=float(np.mean([m["snr_A_db"] for m in ms])),
            snr_B_db=float(np.mean([m["snr_B_db"] for m in ms])),
            xcorr_peak=float(np.mean([m["xcorr_peak"] for m in ms])),
        )
        cell_aggs[(cond, ftype)] = agg
        rows_coh.append(agg)
        rows_snr.append(dict(dataset="XJTU", cell=f"{cond}_{ftype}",
                              channel_A="H", channel_B="V",
                              snr_A_db=agg["snr_A_db"], snr_B_db=agg["snr_B_db"],
                              sig_power_A=float(np.mean([m["sig_power_A"] for m in ms])),
                              noise_power_A=float(np.mean([m["noise_power_A"] for m in ms])),
                              sig_power_B=float(np.mean([m["sig_power_B"] for m in ms])),
                              noise_power_B=float(np.mean([m["noise_power_B"] for m in ms]))))
    return per_bearing, cell_aggs


# ───────────────────────────── gain lookups ──────────────────────────────────

def load_summary(name):
    p = PROJ_ROOT / "results" / name / "summary.json"
    return json.loads(p.read_text())


def cwru_snr_dirname(prefix, snr):
    if snr == 0:
        return prefix  # exp_b2_dual_nokin (no suffix for 0dB base dir)
    return f"{prefix}_snrm{abs(snr)}"


def gather_cwru_gain_points(coherence_by_snr):
    points = []
    for snr, coh in sorted(coherence_by_snr.items(), reverse=True):
        try:
            if snr == 0:
                single = load_summary("exp02_snr0_nokin")
                dual = load_summary("exp_b2_dual_nokin")
            else:
                single = load_summary(f"exp02_snr{snr}_nokin")
                dual = load_summary(f"exp_b2_dual_nokin_snrm{abs(snr)}")
        except FileNotFoundError:
            continue
        gain = dual["mean_best_val_acc"] - single["mean_best_val_acc"]
        points.append(dict(dataset="CWRU", condition=f"SNR={snr}dB",
                            coherence=coh, gain_pp=gain * 100.0))
    return points


def gather_xjtu_gain_points(cells):
    points = []
    # LOBO: Cond3 only (Bearing3_1,3,4,5)
    cond3_cells = [v for (c, f), v in cells.items() if c == "40Hz10kN"]
    if cond3_cells:
        coh_cond3 = float(np.mean([c["mean_coherence_fault_bands"] for c in cond3_cells]))
        single = load_summary("exp_xjtu_lobo_nokin")
        dual = load_summary("exp_xjtu_lobo_dual_nokin")
        gain = dual["mean_macro_recall"] - single["mean_macro_recall"]
        points.append(dict(dataset="XJTU", condition="LOBO(Cond3)",
                            coherence=coh_cond3, gain_pp=gain * 100.0))
    # Cross: Cond2 -> Cond3, use mean coherence over both conditions
    cross_cells = [v for (c, f), v in cells.items() if c in ("37.5Hz11kN", "40Hz10kN")]
    if cross_cells:
        coh_cross = float(np.mean([c["mean_coherence_fault_bands"] for c in cross_cells]))
        single = load_summary("exp_xjtu_cross_nokin")
        dual = load_summary("exp_xjtu_cross_dual_nokin")
        gain = dual["mean_macro_f1"] - single["mean_macro_f1"]
        points.append(dict(dataset="XJTU", condition="Cross(Cond2->Cond3)",
                            coherence=coh_cross, gain_pp=gain * 100.0))
    return points


# ───────────────────────────── main ──────────────────────────────────────────

def main():
    rows_coh, rows_snr = [], []

    print("Running CWRU cells (16 fault_type x load)...")
    cwru_cache = run_cwru(rows_coh, rows_snr)

    print("Running CWRU SNR sweep (coherence at matching SNR levels)...")
    cwru_snr_list = [-8, -6, -4, -2, 0]
    coh_by_snr = run_cwru_snr_sweep(cwru_cache, cwru_snr_list)
    print("  coherence(SNR):", coh_by_snr)

    print("Running XJTU cells (condition x fault_type, 11 bearings)...")
    per_bearing, cells = run_xjtu(rows_coh, rows_snr)

    print("Gathering matched (coherence, gain) points...")
    cwru_points = gather_cwru_gain_points(coh_by_snr)
    xjtu_points = gather_xjtu_gain_points(cells)
    all_points = cwru_points + xjtu_points

    # ── write CSVs ──
    import csv as csvmod

    coh_cols = ["dataset", "fault_type", "load_hp", "condition", "bearing", "rpm",
                "n_files", "n_bearings", "bearings", "n_samples",
                "mean_coherence_fault_bands", "mean_coherence_all",
                "snr_A_db", "snr_B_db", "xcorr_peak", "xcorr_peak_lag_samples", "label"]
    with open(OUT_DIR / "coherence_by_condition.csv", "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=coh_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_coh:
            w.writerow(r)

    snr_cols = ["dataset", "cell", "channel_A", "channel_B",
                "snr_A_db", "snr_B_db", "sig_power_A", "noise_power_A",
                "sig_power_B", "noise_power_B"]
    with open(OUT_DIR / "per_channel_snr.csv", "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=snr_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_snr:
            w.writerow(r)

    with open(OUT_DIR / "gain_vs_coherence_points.csv", "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=["dataset", "condition", "coherence", "gain_pp"])
        w.writeheader()
        for r in all_points:
            w.writerow(r)

    # ── spearman ──
    def spearman_safe(points):
        if len(points) < 3:
            return None, None, len(points)
        xs = [p["coherence"] for p in points]
        ys = [p["gain_pp"] for p in points]
        r, p = spearmanr(xs, ys)
        return float(r), float(p), len(points)

    r_all, p_all, n_all = spearman_safe(all_points)
    r_cwru, p_cwru, n_cwru = spearman_safe(cwru_points)

    print(f"Spearman (all, N={n_all}): r={r_all}, p={p_all}")
    print(f"Spearman (CWRU only, N={n_cwru}): r={r_cwru}, p={p_cwru}")

    # ── plot ──
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    colors = {"CWRU": "#1f77b4", "XJTU": "#d62728"}
    for ds in ("CWRU", "XJTU"):
        pts = [p for p in all_points if p["dataset"] == ds]
        if pts:
            ax.scatter([p["coherence"] for p in pts], [p["gain_pp"] for p in pts],
                       label=ds, color=colors[ds], s=40, zorder=3)
            for p in pts:
                ax.annotate(p["condition"], (p["coherence"], p["gain_pp"]),
                            fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mean coherence in fault-characteristic bands")
    ax.set_ylabel("Dual - single fusion gain (pp)")
    title = f"Fusion gain vs. channel coherence\nSpearman r={r_all:.2f} (N={n_all})" if r_all is not None else "Fusion gain vs. channel coherence"
    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gain_vs_coherence.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "gain_vs_coherence.png", bbox_inches="tight", dpi=200)

    # ── summary.md ──
    lines = []
    lines.append("# A1 channel-coherence analysis summary\n")
    lines.append(f"Output dir: `{OUT_DIR.relative_to(PROJ_ROOT)}`\n")
    lines.append("## Coverage\n")
    n_cwru_cells = sum(1 for r in rows_coh if r["dataset"] == "CWRU")
    n_xjtu_cells = sum(1 for r in rows_coh if r["dataset"] == "XJTU")
    lines.append(f"- CWRU: {n_cwru_cells} (fault_type x load) cells, DE/FE, "
                 f"fault bands from DE_BEARING geometry (project convention).\n")
    lines.append(f"- XJTU: {n_xjtu_cells} (condition x fault_type) cells over "
                 f"{len(per_bearing)} valid OR/IR bearings, H/V, fault-phase windows only.\n")

    lines.append("\n## Fault-band coherence, by dataset\n")
    for ds in ("CWRU", "XJTU"):
        vals = [r["mean_coherence_fault_bands"] for r in rows_coh if r["dataset"] == ds]
        lines.append(f"- {ds}: mean={np.mean(vals):.3f}, min={np.min(vals):.3f}, "
                     f"max={np.max(vals):.3f} across {len(vals)} cells.\n")

    cwru_normal = [r for r in rows_coh if r["dataset"] == "CWRU" and r["fault_type"] == "normal"]
    cwru_fault = [r for r in rows_coh if r["dataset"] == "CWRU" and r["fault_type"] != "normal"]
    if cwru_normal and cwru_fault:
        lines.append(f"\n- Sanity control: CWRU `normal` cells (no seeded fault) mean "
                     f"fault-band SNR_A={np.mean([r['snr_A_db'] for r in cwru_normal]):.2f} dB "
                     f"vs faulted cells mean SNR_A={np.mean([r['snr_A_db'] for r in cwru_fault]):.2f} dB.\n")

    lines.append("\n## Gain vs. coherence (task 3)\n")
    lines.append("Matched (coherence, gain) pairs -- see `gain_vs_coherence_points.csv` "
                 "and the design-notes docstring in this script for exactly how each point "
                 "was constructed (granularity mismatch between available gain data and "
                 "computed coherence cells forced this specific pairing; documented, not hidden).\n")
    for p in all_points:
        lines.append(f"- {p['dataset']} / {p['condition']}: coherence={p['coherence']:.3f}, "
                     f"gain={p['gain_pp']:+.2f} pp\n")
    if r_all is not None:
        lines.append(f"\n**Spearman (all {n_all} points): r={r_all:.3f}, p={p_all:.3f}.**\n")
    if r_cwru is not None:
        lines.append(f"**Spearman (CWRU-only SNR sweep, {n_cwru} points): "
                     f"r={r_cwru:.3f}, p={p_cwru:.3f}.**\n")
    lines.append("\nN is very small (5-7 points, 2 of them from a different dataset/metric "
                 "than the other 5); this is directional evidence only, not a confirmed "
                 "relationship. No claim beyond what these numbers show is made.\n")

    (OUT_DIR / "a1_summary.md").write_text("".join(lines), encoding="utf-8")
    print("Done. Wrote:", list(OUT_DIR.glob("*")))


if __name__ == "__main__":
    main()
