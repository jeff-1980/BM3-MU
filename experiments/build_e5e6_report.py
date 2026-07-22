#!/usr/bin/env python3
"""
experiments/build_e5e6_report.py

Read-only assembly of the E5 (pink noise) + E6 (flagship new-seed extension)
results into a vault-page appendix. No training, no existing-file edits.

E5: results/exp_e5_{single,dual}_pink_snr{-4,-6,-8}/summary.json
    (best_val_acc mean+-std per arm x SNR, pink noise, 5 seeds)

E6: merges OLD seeds {0..4} (existing flagship results) with NEW seeds {5,6,7}
    (results/exp_e6_<arm>_snrm<N>_newseed/summary.json) into n=8 per arm x SNR,
    then runs an exact paired Wilcoxon signed-rank test (dual vs single, matched
    by seed) for both the nokin and kin variant at each SNR.

Usage:
  python build_e5e6_report.py --append-to /path/to/vault/回传-加固实验.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJ_ROOT = Path(__file__).resolve().parent.parent
RES = PROJ_ROOT / "results"
SNRS = [-4, -6, -8]


def load_summary(name):
    p = RES / name / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def snr_tag_m(snr):
    return f"snrm{abs(snr)}"


# ─────────────────────────── E5 ──────────────────────────────────────────────

def build_e5_section():
    lines = ["## E5 — Pink noise (CWRU, dual vs single CE-only)\n\n",
             "噪声生成：`bearmamba3/noise.py::generate_pink_noise()`（FFT 幅度谱按 "
             "1/sqrt(f) 缩放 → 功率谱 ~1/f），谱验证：`results/figures/e5_pink_noise_"
             "spectrum_verification.png`（拟合斜率 -0.998，白噪声对照斜率 +0.000，"
             "均已通过脚本内断言）。\n\n",
             "| SNR | single (CE-only) best_val_acc | dual (CE-only) best_val_acc | dual-single Δ (pp) |\n",
             "|---|---|---|---|\n"]
    for snr in SNRS:
        s_single = load_summary(f"exp_e5_single_pink_snr{snr}")
        s_dual = load_summary(f"exp_e5_dual_pink_snr{snr}")
        if s_single is None or s_dual is None:
            lines.append(f"| {snr} | n/a | n/a | n/a |\n")
            continue
        single_v = f"{s_single['mean_best_val_acc']*100:.2f}±{s_single['std_best_val_acc']*100:.2f}"
        dual_v = f"{s_dual['mean_best_val_acc']*100:.2f}±{s_dual['std_best_val_acc']*100:.2f}"
        delta = (s_dual['mean_best_val_acc'] - s_single['mean_best_val_acc']) * 100
        lines.append(f"| {snr} | {single_v} | {dual_v} | {delta:+.2f} |\n")
    lines.append("\n")
    return "".join(lines)


# ─────────────────────────── E6 ──────────────────────────────────────────────

OLD_SOURCE = {
    "single_nokin": lambda snr: f"exp02_snr{snr}_nokin",
    "single_kin": lambda snr: f"exp02_snr{snr}_kin",
    "dual_nokin": lambda snr: f"exp_b2_dual_nokin_{snr_tag_m(snr)}",
    "dual_kin": lambda snr: f"exp_b2_dual_kin_{snr_tag_m(snr)}",
}
NEW_SOURCE = {
    "single_nokin": lambda snr: f"exp_e6_single_nokin_{snr_tag_m(snr)}_newseed",
    "single_kin": lambda snr: f"exp_e6_single_kin_{snr_tag_m(snr)}_newseed",
    "dual_nokin": lambda snr: f"exp_e6_dual_nokin_{snr_tag_m(snr)}_newseed",
    "dual_kin": lambda snr: f"exp_e6_dual_kin_{snr_tag_m(snr)}_newseed",
}


def merged_seed_dict(arm, snr):
    """Return {seed: best_val_acc} merging old (0-4) + new (5-7) seeds, or None if incomplete."""
    old = load_summary(OLD_SOURCE[arm](snr))
    new = load_summary(NEW_SOURCE[arm](snr))
    if old is None or new is None:
        return None
    d = dict(zip(old["seeds"], old["best_val_accs"]))
    d.update(dict(zip(new["seeds"], new["best_val_accs"])))
    return d


def build_e6_section():
    lines = ["## E6 — Flagship cells, new-seed extension (n=8, seeds 0-4 + 5-7)\n\n",
             "旧 seed(0-4）来自既有结果目录，未重跑；新 seed(5-7）来自 "
             "`results/exp_e6_<arm>_snrm<N>_newseed/`。合并后逐 arm×SNR n=8。\n\n",
             "| SNR | arm | n | mean best_val_acc ± std |\n|---|---|---|---|\n"]
    merged = {}
    for snr in SNRS:
        for arm in ("single_nokin", "single_kin", "dual_nokin", "dual_kin"):
            d = merged_seed_dict(arm, snr)
            merged[(arm, snr)] = d
            if d is None:
                lines.append(f"| {snr} | {arm} | n/a | n/a |\n")
                continue
            vals = np.array(list(d.values()))
            lines.append(f"| {snr} | {arm} | {len(vals)} | {vals.mean()*100:.2f}±{vals.std(ddof=1)*100:.2f} |\n")
    lines.append("\n")

    lines.append("### dual − single 配对（同 seed 匹配，n=8），exact Wilcoxon signed-rank\n\n")
    lines.append("| SNR | 变体 | mean Δ(dual-single) pp | 8/8 同向？ | Wilcoxon W | p (exact) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for snr in SNRS:
        for variant, single_arm, dual_arm in (("nokin", "single_nokin", "dual_nokin"),
                                                ("kin", "single_kin", "dual_kin")):
            d_single = merged.get((single_arm, snr))
            d_dual = merged.get((dual_arm, snr))
            if d_single is None or d_dual is None:
                lines.append(f"| {snr} | {variant} | n/a | n/a | n/a | n/a |\n")
                continue
            common_seeds = sorted(set(d_single) & set(d_dual))
            single_vals = [d_single[s] for s in common_seeds]
            dual_vals = [d_dual[s] for s in common_seeds]
            diffs = [d - s for d, s in zip(dual_vals, single_vals)]
            mean_delta = float(np.mean(diffs)) * 100
            n_pos = sum(1 for x in diffs if x > 0)
            n_neg = sum(1 for x in diffs if x < 0)
            concordant = f"{max(n_pos, n_neg)}/{len(diffs)}"
            try:
                stat, p = wilcoxon(dual_vals, single_vals, mode="exact")
            except ValueError:
                stat, p = float("nan"), float("nan")
            lines.append(f"| {snr} | {variant} | {mean_delta:+.2f} | {concordant} | {stat} | {p} |\n")
    lines.append("\n")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append-to", required=True)
    args = ap.parse_args()

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    out = [f"\n---\n\n# 回传：E5 + E6（{now}）\n\n",
           build_e5_section(),
           build_e6_section(),
           "不下结论，判据留人工裁决。\n"]

    path = Path(args.append_to)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + "".join(out), encoding="utf-8")
    print(f"Appended E5+E6 section to {path}")


if __name__ == "__main__":
    main()
