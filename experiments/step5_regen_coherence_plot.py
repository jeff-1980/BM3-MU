#!/usr/bin/env python3
"""
Regenerate the A1 gain-vs-coherence scatter (S1, supplementary material)
with the day-2 text-surgery updated gain values: CWRU -4/-6/-8dB use the
n=8 merged E6 gains (Section subsubsec:b2); XJTU LOBO uses the leakage-free
CE-only headline gain (-18.9pp) instead of the retired original-protocol
number (-15.53pp). Coherence values are unchanged (intrinsic signal
property, independent of epoch-selection protocol).

Does not overwrite results/a1_coherence_20260708-2358/gain_vs_coherence.*
(results/ append-only guardrail). Writes new files under
results/figures/gain_vs_coherence_n8.{png,pdf} and a new CSV alongside.
"""
import csv
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

points = [
    ("CWRU", "SNR=0dB",   0.013335292227566242,  0.14),
    ("CWRU", "SNR=-2dB",  0.008194568566977978,  0.05),
    ("CWRU", "SNR=-4dB",  0.0038763377815485,    0.18),   # n=8 updated (was 0.70)
    ("CWRU", "SNR=-6dB",  0.0014647903153672814, 0.92),   # n=8 updated (was 1.75)
    ("CWRU", "SNR=-8dB",  0.0008575583924539387, 2.20),   # n=8 updated (was 2.53)
    ("XJTU", "LOBO(Cond3)", 0.41167497634887695, -18.9),  # leakage-free (was -15.53)
    ("XJTU", "Cross(Cond2->Cond3)", 0.4285101542870204, 14.01),
]

new_csv = OUT_DIR.parent / "a1_coherence_20260708-2358" / "gain_vs_coherence_points_n8.csv"
with open(new_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "condition", "coherence", "gain_pp"])
    for row in points:
        w.writerow(row)

cwru = [(c, g) for ds, cond, c, g in points if ds == "CWRU"]
xjtu = [(c, g) for ds, cond, c, g in points if ds == "XJTU"]

coh_cwru = [c for c, g in cwru]
gain_cwru = [g for c, g in cwru]
r5, p5 = stats.spearmanr(coh_cwru, gain_cwru)

coh_all = [c for c, g in points[:0]] or [p[2] for p in points]
gain_all = [p[3] for p in points]
r7, p7 = stats.spearmanr(coh_all, gain_all)
print(f"CWRU-only (n=5): r={r5:.3f}, p={p5:.3f}")
print(f"Pooled (n=7): r={r7:.3f}, p={p7:.3f}")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.scatter(coh_cwru, gain_cwru, color="#1f77b4", label="CWRU (AWGN sweep)", zorder=3)
ax.scatter([c for c, g in xjtu], [g for c, g in xjtu], color="#d62728",
           marker="s", label="XJTU-SY (natural)", zorder=3)
for ds, cond, c, g in points:
    ax.annotate(cond, (c, g), fontsize=6, xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="#aaaaaa", lw=0.7, ls=":")
ax.set_xlabel("Fault-band channel coherence")
ax.set_ylabel("Dual $-$ single gain (pp)")
ax.set_title(f"Gain vs. coherence (updated, n=8 CWRU + leak-free LOBO)\n"
             f"CWRU-only Spearman r={r5:.3f}, p={p5:.3f}; pooled r={r7:.3f}, p={p7:.3f}",
             fontsize=7.5)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "gain_vs_coherence_n8.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "gain_vs_coherence_n8.pdf", bbox_inches="tight")
print("Saved -> gain_vs_coherence_n8.{png,pdf}")
