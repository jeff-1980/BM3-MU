#!/usr/bin/env python3
"""
experiments/exp_e5_pink_noise/verify_pink_spectrum.py

Spectral verification for bearmamba3.noise.generate_pink_noise(): confirms the
generated noise actually has a 1/f power spectral density (log-log PSD slope
~= -1), as opposed to AWGN's flat spectrum. Read-only / generates a plot only,
no training.

Usage:
  source venv/bin/activate
  python experiments/exp_e5_pink_noise/verify_pink_spectrum.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))
from bearmamba3.noise import generate_pink_noise

OUT_DIR = PROJ_ROOT / "results" / "figures"
FS = 12000  # CWRU sampling rate, for realistic frequency-axis labelling
N = 12000 * 20  # 20s worth of samples, long enough for a stable Welch estimate


def main():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(N).astype(np.float32)
    pink = generate_pink_noise(N, rng)

    f_w, Pxx_w = scipy.signal.welch(white, fs=FS, nperseg=4096)
    f_p, Pxx_p = scipy.signal.welch(pink, fs=FS, nperseg=4096)

    # fit log-log slope over the mid-band (skip DC bin and the very top decade
    # where Welch variance dominates)
    mask = (f_p > 5) & (f_p < FS / 2 * 0.8)
    slope_p, intercept_p = np.polyfit(np.log10(f_p[mask]), np.log10(Pxx_p[mask]), 1)
    slope_w, intercept_w = np.polyfit(np.log10(f_w[mask]), np.log10(Pxx_w[mask]), 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=150)
    ax.loglog(f_w[1:], Pxx_w[1:], label=f"white noise (fit slope={slope_w:+.2f})",
              color="#7f7f7f", alpha=0.8, lw=1)
    ax.loglog(f_p[1:], Pxx_p[1:], label=f"pink noise (fit slope={slope_p:+.2f}, target=-1.00)",
              color="#d62728", lw=1.2)
    ref = Pxx_p[mask][0] * (f_p[mask][0] / f_p[mask]) ** 1.0
    ax.loglog(f_p[mask], ref, "--", color="black", lw=1, alpha=0.6, label="ideal 1/f reference")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (Welch, nperseg=4096)")
    ax.set_title("Pink vs white noise PSD — spectral verification (bearmamba3.noise)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e5_pink_noise_spectrum_verification.png", bbox_inches="tight", dpi=200)
    fig.savefig(OUT_DIR / "e5_pink_noise_spectrum_verification.pdf", bbox_inches="tight")

    print(f"pink noise fit slope:  {slope_p:+.3f}  (ideal 1/f = -1.00)")
    print(f"white noise fit slope: {slope_w:+.3f}  (ideal white = 0.00)")
    assert -1.3 < slope_p < -0.7, f"pink noise slope {slope_p:.3f} not close to -1 — generator is broken"
    assert -0.3 < slope_w < 0.3, f"white noise slope {slope_w:.3f} not close to 0 — Welch/setup is broken"
    print("PASS: pink noise slope close to -1, white noise slope close to 0.")
    print(f"Saved: {OUT_DIR / 'e5_pink_noise_spectrum_verification.png'}")


if __name__ == "__main__":
    main()
