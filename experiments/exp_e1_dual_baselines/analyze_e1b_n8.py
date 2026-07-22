#!/usr/bin/env python3
"""
experiments/exp_e1_dual_baselines/analyze_e1b_n8.py

E1b n=8 extension (third-round review, M1) — read-only merge + analysis.
Merges each of the six arms' existing seed-0-4 summary.json with the new
seed-5-7 summary.json into an n=8 pool (paired by seed id, not position),
computes per-arm n=8 mean+-std, and for each of the three backbones runs
an exact two-sided Wilcoxon signed-rank test on the dual-single pairing.

Per prereg_e1b_n8.md: report is unconditional. If any backbone's pairing
loses 8/8 concordance, that is reported as-is (narrated as a downgrade),
not hidden or re-run.

No existing results/ file is modified; this script only reads summary.json
files and writes a new report to the vault path below.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
VAULT_OUT = Path(
    "/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/管家/回传-e1b-n8.md"
)

# (arm label, backbone, sensor config, old summary.json, new summary.json)
ARMS = [
    ("single-CNN", "cnn1d", "single",
     ROOT / "results/exp_mext_e13_1dcnn_xjtu_cross/summary.json",
     ROOT / "results/exp_mext_e13_1dcnn_xjtu_cross_newseed/summary.json"),
    ("dual-CNN", "cnn1d", "dual",
     ROOT / "results/e1b_fillcells_20260709-1515/xjtu_cross_dual_cnn/summary.json",
     ROOT / "results/exp_e1b_xjtu_cross_dual_cnn_newseed/summary.json"),
    ("single-BM2", "bm2", "single",
     ROOT / "results/e1b_fillcells_20260709-1515/xjtu_cross_single_bm2/summary.json",
     ROOT / "results/exp_e1b_xjtu_cross_single_bm2_newseed/summary.json"),
    ("dual-BM2", "bm2", "dual",
     ROOT / "results/e1_dual_baselines_20260709-1052/xjtu_cross_dual_bm2/summary.json",
     ROOT / "results/exp_e1_xjtu_cross_dual_bm2_newseed/summary.json"),
    ("single-BM3 (CE-only)", "bm3", "single",
     ROOT / "results/exp_xjtu_cross_nokin/summary.json",
     ROOT / "results/exp_xjtu_cross_nokin_newseed/summary.json"),
    ("dual-BM3 (CE-only)", "bm3", "dual",
     ROOT / "results/exp_xjtu_cross_dual_nokin/summary.json",
     ROOT / "results/exp_xjtu_cross_dual_nokin_newseed/summary.json"),
]


def load_seed_f1(path):
    d = json.loads(Path(path).read_text())
    seeds = d["seeds"]
    f1s = d["macro_f1s"]
    return dict(zip(seeds, f1s))


def merged_n8(old_path, new_path):
    old = load_seed_f1(old_path)
    new = load_seed_f1(new_path)
    merged = {**old, **new}
    seeds_sorted = sorted(merged.keys())
    vals = np.array([merged[s] for s in seeds_sorted]) * 100
    return seeds_sorted, vals


def main():
    arm_data = {}
    lines = []
    lines.append("# 回传：E1b n=8 扩展（第三轮评审 M1，2026-07-21）\n")
    lines.append(
        "预注册：`prereg_e1b_n8.md`（写入时间早于起跑，锁定统计裁决规则："
        "三组 dual-single 配对做 exact 两侧 Wilcoxon，如实报告，任何一组失去 "
        "8/8 同向即在正文降级措辞，不回避不换指标）。\n"
    )
    lines.append("## 六臂 n=8 合并结果（macro-F1, %）\n")
    lines.append("| 臂 | seeds 合并 | mean ± std (n=8) |")
    lines.append("|---|---|---|")

    for label, backbone, sensor, old_p, new_p in ARMS:
        seeds_sorted, vals = merged_n8(old_p, new_p)
        arm_data[(backbone, sensor)] = (seeds_sorted, vals)
        mean, std = vals.mean(), vals.std(ddof=1)
        lines.append(
            f"| {label} | {seeds_sorted} | {mean:.2f} ± {std:.2f} |"
        )

    lines.append("")
    lines.append("## 三组 dual − single 配对（exact 两侧 Wilcoxon, n=8）\n")
    lines.append("| 骨干 | seeds | single (per-seed) | dual (per-seed) | diff (dual-single) | 同向计数 | Wilcoxon p (two-sided, exact) |")
    lines.append("|---|---|---|---|---|---|---|")

    narrative_flags = []
    for backbone, backbone_label in [("cnn1d", "1D-CNN"), ("bm2", "BearMamba-2"), ("bm3", "BearMamba-3 (CE-only)")]:
        seeds_s, vals_s = arm_data[(backbone, "single")]
        seeds_d, vals_d = arm_data[(backbone, "dual")]
        assert seeds_s == seeds_d, f"seed mismatch for {backbone}: {seeds_s} vs {seeds_d}"
        diffs = vals_d - vals_s
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        n = len(diffs)
        concordant = max(n_pos, n_neg)
        try:
            w = stats.wilcoxon(vals_d, vals_s, alternative="two-sided", mode="exact")
            pval = w.pvalue
        except Exception as e:
            pval = float("nan")
        concordance_str = f"{concordant}/{n}"
        if concordant < n:
            narrative_flags.append(
                f"{backbone_label}: {concordance_str} 同向（非全同向，n=8 下方向一致性较 n=5 减弱），"
                f"exact two-sided p={pval:.4f}"
            )
        else:
            narrative_flags.append(
                f"{backbone_label}: {concordance_str} 同向（维持全同向），"
                f"exact two-sided p={pval:.4f}"
            )
        diff_str = ", ".join(f"{d:+.2f}" for d in diffs)
        single_str = ", ".join(f"{v:.2f}" for v in vals_s)
        dual_str = ", ".join(f"{v:.2f}" for v in vals_d)
        lines.append(
            f"| {backbone_label} | {seeds_s} | {single_str} | {dual_str} | {diff_str} | "
            f"{concordance_str} | {pval:.4f} |"
        )

    lines.append("")
    lines.append("## 如实结论（按预注册规则，不做取舍）\n")
    for flag in narrative_flags:
        lines.append(f"- {flag}")
    lines.append("")
    any_downgrade = any("非全同向" in f for f in narrative_flags)
    if any_downgrade:
        lines.append(
            "**判定：至少一组在 n=8 下失去 8/8 同向 → 按预注册规则，正文对应措辞"
            "须降级，不得再用'5/5(或8/8) seeds concordant'的全同向表述，"
            "改用实际同向计数 + 对应 p 值，并如实注明方向一致性减弱。**"
        )
    else:
        lines.append(
            "**判定：三组均维持全同向（8/8）→ 沿用现有'三骨干独立复现同方向'叙事，"
            "仅把 n=5 数字与 p 值换成本报告的 n=8 数字（正文 Table "
            "tab:e1b_backbone_agnostic 及相关表述待人工据此更新）。**"
        )
    lines.append("")
    lines.append(
        "与 CWRU E6 压力测试处理逻辑对称：本报告只如实呈现 n=8 合并结果，"
        "不在此脚本内自动改写论文正文，措辞降级/保留由人工据上表确认后执行。"
    )

    report = "\n".join(lines) + "\n"
    VAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    VAULT_OUT.write_text(report, encoding="utf-8")
    print(f"Wrote report to {VAULT_OUT}")
    print()
    print(report)


if __name__ == "__main__":
    main()
