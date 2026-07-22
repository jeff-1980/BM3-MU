#!/usr/bin/env python3
"""
Step 5 / B2 (n=8 update, 2026-07-20 text-surgery day 1, T2):
Single vs dual-sensor SNR curve, merging original 5 seeds with the
E6 new-seed extension (seeds 5-7) at the three "flagship" SNR points
(-4/-6/-8 dB). -2/0 dB remain n=5 (E6 did not extend those points).

Supersedes the n=5-only annotation in step5_b2_snr_curve.py: the former
"-4dB nokin Wilcoxon p=0.031 (unique significant point)" callout does not
survive the n=8 re-check (p=0.64) and is removed here; replaced with a
monotonic-trend annotation, consistent with the CLAUDE.md / vault
E5/E6 human read-out (2026-07-20).

Does NOT overwrite results/figures/b2_snr_curve_dual.{pdf,png} (results/
guardrail: append-only). Output is a new file,
results/figures/b2_snr_curve_dual_n8.{pdf,png}.
"""
import json
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT    = pathlib.Path(__file__).resolve().parent.parent / "results"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNRS = [-8, -6, -4, -2, 0]         # +10dB excluded (single already 100%)
N8_SNRS = {-8, -6, -4}             # E6 new-seed extension only covers these


def load_accs(exp_dir: pathlib.Path):
    p = exp_dir / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    accs = np.array(d.get("best_val_accs", [])) * 100
    return accs if len(accs) else None


def get_old_dual_dir(kind: str, snr: int) -> pathlib.Path:
    if snr == 0:
        return ROOT / f"exp_b2_dual_{kind}"
    tag = f"snrm{abs(snr)}" if snr < 0 else f"snr{snr}"
    return ROOT / f"exp_b2_dual_{kind}_{tag}"


def get_old_single_dir(kind: str, snr: int) -> pathlib.Path:
    return ROOT / f"exp02_snr{snr}_{kind}"


def get_e6_dir(arm: str, kind: str, snr: int) -> pathlib.Path:
    return ROOT / f"exp_e6_{arm}_{kind}_snrm{abs(snr)}_newseed"


def merged_seeds(arm: str, kind: str, snr: int):
    """arm='single'|'dual'. Returns concatenated seed-accuracy array (n=5 or n=8)."""
    old_dir = get_old_single_dir(kind, snr) if arm == "single" else get_old_dual_dir(kind, snr)
    old = load_accs(old_dir)
    if old is None:
        return None
    if snr in N8_SNRS:
        new = load_accs(get_e6_dir(arm, kind, snr))
        if new is not None:
            return np.concatenate([old, new])
    return old


# ── load merged data ──────────────────────────────────────────────────────────
data = {}
for cond in ("nokin", "kin"):
    data[cond] = {"single": {"seeds": [], "mean": [], "std": [], "n": []},
                  "dual":   {"seeds": [], "mean": [], "std": [], "n": []}}
    for snr in SNRS:
        for arm in ("single", "dual"):
            seeds = merged_seeds(arm, cond, snr)
            data[cond][arm]["seeds"].append(seeds)
            data[cond][arm]["mean"].append(float(seeds.mean()) if seeds is not None else np.nan)
            data[cond][arm]["std"].append(float(seeds.std(ddof=1)) if seeds is not None else np.nan)
            data[cond][arm]["n"].append(len(seeds) if seeds is not None else 0)

print("Sample sizes per SNR (single/dual, nokin):",
      list(zip(SNRS, data["nokin"]["single"]["n"], data["nokin"]["dual"]["n"])))

# Wilcoxon (dual vs single, per SNR, paired by seed index — valid for the
# n=8 points because seeds [0..4] are shared across old runs and [5..7] are
# shared across the E6 new-seed batch; for n=5 points this reduces to the
# original pairing).
p_nokin, p_kin = [], []
for i, snr in enumerate(SNRS):
    sn, dn = data["nokin"]["single"]["seeds"][i], data["nokin"]["dual"]["seeds"][i]
    sk, dk = data["kin"]["single"]["seeds"][i], data["kin"]["dual"]["seeds"][i]
    try:
        p_nokin.append(stats.wilcoxon(dn, sn, alternative="greater").pvalue)
    except Exception:
        p_nokin.append(float("nan"))
    try:
        p_kin.append(stats.wilcoxon(dk, sk, alternative="greater").pvalue)
    except Exception:
        p_kin.append(float("nan"))

print("B2(n8) Wilcoxon p-values (dual > single, one-sided):")
for snr, pn, pk, n in zip(SNRS, p_nokin, p_kin, data["nokin"]["dual"]["n"]):
    print(f"  {snr:+3d}dB (n={n})  nokin p={pn:.4f}  kin p={pk:.4f}")

snrs = np.array(SNRS, dtype=float)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                          gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.38})

COLOR = {"nokin": "#1f77b4", "kin": "#2ca02c"}
LABEL = {"nokin": "CE-only", "kin": "+L_kin"}

# ── panel (a): accuracy curves ────────────────────────────────────────────────
ax = axes[0]
ax.set_title("(a) Single vs Dual Sensor — Accuracy vs SNR (n=8 @ -4/-6/-8dB)")

for cond in ("nokin", "kin"):
    c = COLOR[cond]
    lbl = LABEL[cond]
    m_s = np.array(data[cond]["single"]["mean"])
    s_s = np.array(data[cond]["single"]["std"])
    m_d = np.array(data[cond]["dual"]["mean"])
    s_d = np.array(data[cond]["dual"]["std"])

    ax.fill_between(snrs, m_s - s_s, m_s + s_s, alpha=0.18, color=c)
    ax.plot(snrs, m_s, color=c, lw=1.4, ls="--",
            marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label=f"Single {lbl}")
    ax.fill_between(snrs, m_d - s_d, m_d + s_d, alpha=0.30, color=c)
    ax.plot(snrs, m_d, color=c, lw=2.0, ls="-",
            marker="o", markersize=5,
            label=f"Dual {lbl}")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(-9.5, 1.5)
ax.set_xticks(SNRS)
ax.set_xticklabels([f"{s:+d}" for s in SNRS])
ax.set_ylim(82, 101.5)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(axis="y", ls=":", alpha=0.45)
ax.grid(axis="x", ls=":", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.88, ncol=2)

# ── panel (b): gain (dual - single), monotonic-trend annotation (no
#     significance claim — replaces the retired p=0.031 callout) ─────────────
ax2 = axes[1]
ax2.set_title("(b) Gain: Dual − Single (pp)")

for cond in ("nokin", "kin"):
    c = COLOR[cond]
    lbl = LABEL[cond]
    gain = np.array(data[cond]["dual"]["mean"]) - np.array(data[cond]["single"]["mean"])
    ax2.plot(snrs, gain, color=c, lw=1.8, ls="-" if cond == "nokin" else "--",
             marker="o", markersize=5,
             label=f"{lbl}")
    for i, (snr, g) in enumerate(zip(SNRS, gain)):
        if abs(g) > 0.3:
            ax2.text(snr + 0.15, g + 0.05, f"+{g:.2f}", fontsize=6.5,
                     color=c, va="bottom")

ax2.axhline(0, color="#aaaaaa", lw=0.8, ls=":")
ax2.annotate(
    "monotonic trend with SNR\n(n=8; not significant, see Table)",
    xy=(-6, 1.0), xytext=(-9.2, 2.6),
    fontsize=7, color="#444444",
    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#888888"),
)
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Accuracy gain (pp)")
ax2.set_xlim(-9.5, 1.5)
ax2.set_xticks(SNRS)
ax2.set_xticklabels([f"{s:+d}" for s in SNRS])
ax2.set_ylim(bottom=-0.6)
ax2.grid(axis="y", ls=":", alpha=0.45)
ax2.legend(loc="upper left", framealpha=0.88)

fig.suptitle(
    "CWRU SISO→Dual (DE+FE) / BearMamba-3 — n=8 at -4/-6/-8dB, n=5 at -2/0dB",
    fontsize=8, color="#444444"
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_stem = "b2_snr_curve_dual_n8"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"\nSaved -> {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)
