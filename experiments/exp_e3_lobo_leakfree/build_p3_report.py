#!/usr/bin/env python3
"""
experiments/exp_e3_lobo_leakfree/build_p3_report.py

Read-only P3 adjudication data assembly for the E3 LOBO leak-free protocol.
Called by report_to_vault.sh once all 80 (4 folds x 4 arms x 5 seeds) leakfree
runs are on disk. Never trains anything, never writes under results/.

Compares, per fold x arm:
  - leakfree fixed-epoch protocol (results/e3_lobo_leakfree_<ts>/<arm>/fold{f}_seed{s}.json,
    fields: macro_recall / macro_f1)
  - original per-epoch-selection protocol (results/exp_xjtu_lobo_<arm>/fold{f}_seed{s}.json,
    fields: best_macro_recall / best_macro_f1)

Then, per fold, computes dual-single deltas for BOTH the nokin and kin variants
under each protocol, checks whether the sign (direction) agrees between
protocols, and reports |delta_leakfree - delta_original| as the drift in pp.

Usage:
  python build_p3_report.py --leakfree-dir results/e3_lobo_leakfree_<ts> \
      --out /path/to/vault/回传-加固实验.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent

# leakfree arm subdir name -> original-protocol results dir name
ARM_MAP = {
    "single_nokin": "exp_xjtu_lobo_nokin",
    "single_kin": "exp_xjtu_lobo_kin",
    "dual_nokin": "exp_xjtu_lobo_dual_nokin",
    "dual_kin": "exp_xjtu_lobo_dual_kin",
}
FOLDS = [0, 1, 2, 3]
SEEDS = [0, 1, 2, 3, 4]
FOLD_BEARING = {0: "Bearing3_1(OR)", 1: "Bearing3_3(IR)", 2: "Bearing3_4(IR)", 3: "Bearing3_5(OR)"}


def load_fold_seed(dirpath, fold, seed, recall_key, f1_key):
    p = Path(dirpath) / f"fold{fold}_seed{seed}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d[recall_key], d[f1_key]


def fold_mean(dirpath, fold, recall_key, f1_key):
    recalls, f1s = [], []
    for seed in SEEDS:
        r = load_fold_seed(dirpath, fold, seed, recall_key, f1_key)
        if r is None:
            return None
        recalls.append(r[0])
        f1s.append(r[1])
    return dict(
        mean_recall=float(np.mean(recalls)), std_recall=float(np.std(recalls, ddof=1)),
        mean_f1=float(np.mean(f1s)), std_f1=float(np.std(f1s, ddof=1)),
        n=len(recalls),
    )


def count_and_eta(leakfree_root):
    """Return (n_done, n_total=80, eta_seconds_remaining or None)."""
    n_done = 0
    elapsed_by_arm = {}
    for arm in ARM_MAP:
        arm_dir = leakfree_root / arm
        elapsed = []
        for fold in FOLDS:
            for seed in SEEDS:
                p = arm_dir / f"fold{fold}_seed{seed}.json"
                if p.exists():
                    n_done += 1
                    try:
                        elapsed.append(json.loads(p.read_text()).get("elapsed_s"))
                    except Exception:
                        pass
        elapsed = [e for e in elapsed if e]
        elapsed_by_arm[arm] = elapsed

    # historical fallback (e3_cost_estimate.md mean elapsed_s per run, old protocol)
    HIST_FALLBACK = {"single_nokin": 259.9, "single_kin": 297.0,
                     "dual_nokin": 228.1, "dual_kin": 278.7}
    remaining_s = 0.0
    any_estimate = False
    for arm in ARM_MAP:
        n_arm_done = len(elapsed_by_arm[arm])
        n_arm_remaining = 20 - n_arm_done
        if n_arm_remaining <= 0:
            continue
        if elapsed_by_arm[arm]:
            per_run = float(np.mean(elapsed_by_arm[arm]))
        else:
            per_run = HIST_FALLBACK[arm]
        remaining_s += per_run * n_arm_remaining
        any_estimate = True
    return n_done, 80, (remaining_s if any_estimate else None)


def write_status(leakfree_root, out_path):
    n_done, n_total, remaining_s = count_and_eta(leakfree_root)
    import time as _time
    now = _time.strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# 回传：E3 LOBO 无泄漏协议 — 状态\n\n",
             "**训练进行中**，最后更新：", now, "\n\n",
             f"当前完成：{n_done}/{n_total}\n\n"]
    if remaining_s is not None:
        h = int(remaining_s // 3600)
        m = int((remaining_s % 3600) // 60)
        lines.append(f"预计剩余：约 {h} 小时 {m} 分钟\n\n")
    else:
        lines.append("预计剩余：暂无法估算（尚无已完成 run 的 elapsed_s 可用于外推）\n\n")
    lines.append("（完整 P3 裁决报告将在 80/80 全部完成后生成，届时会替换本页全部内容）\n")
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"[status] {n_done}/{n_total} written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leakfree-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--completed-ts", default=None)
    ap.add_argument("--mode", choices=["status", "report"], default="report")
    args = ap.parse_args()

    leakfree_root = Path(args.leakfree_dir)
    if not leakfree_root.is_absolute():
        leakfree_root = PROJECT_ROOT / leakfree_root

    if args.mode == "status":
        write_status(leakfree_root, args.out)
        return

    # per fold x arm x protocol table
    data = {}  # (arm, fold) -> {"leakfree": {...}, "original": {...}}
    missing = []
    for arm, orig_name in ARM_MAP.items():
        leakfree_dir = leakfree_root / arm
        orig_dir = PROJECT_ROOT / "results" / orig_name
        for fold in FOLDS:
            lf = fold_mean(leakfree_dir, fold, "macro_recall", "macro_f1")
            og = fold_mean(orig_dir, fold, "best_macro_recall", "best_macro_f1")
            if lf is None:
                missing.append(f"{arm} fold{fold} (leakfree)")
            if og is None:
                missing.append(f"{arm} fold{fold} (original)")
            data[(arm, fold)] = {"leakfree": lf, "original": og}

    import time as _time
    completed_ts = args.completed_ts or _time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# 回传：E3 LOBO 无泄漏协议 — P3 裁决数据\n\n")
    lines.append(f"完成时间戳：{completed_ts}\n\n")
    lines.append(f"leakfree 数据源：`{leakfree_root.relative_to(PROJECT_ROOT)}`（fixed_epoch，无逐 epoch 选点）\n\n")
    lines.append("原协议数据源：`results/exp_xjtu_lobo_{nokin,kin,dual_nokin,dual_kin}/`（逐 epoch 测试折选点，历史产物，未改动）\n\n")

    if missing:
        lines.append(f"⚠️ 缺失数据点（{len(missing)} 处，以下比较跳过对应 fold/arm）：{', '.join(missing)}\n\n")

    lines.append("## 逐折 × arm：leakfree(fixed_epoch) vs 原协议(best_epoch)，macro_recall / macro_f1（5 seed 均值±std）\n\n")
    lines.append("| fold | test bearing | arm | leakfree recall | 原协议 recall | leakfree F1 | 原协议 F1 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for fold in FOLDS:
        for arm in ARM_MAP:
            d = data[(arm, fold)]
            lf, og = d["leakfree"], d["original"]
            lf_r = f"{lf['mean_recall']*100:.2f}±{lf['std_recall']*100:.2f}" if lf else "n/a"
            og_r = f"{og['mean_recall']*100:.2f}±{og['std_recall']*100:.2f}" if og else "n/a"
            lf_f = f"{lf['mean_f1']*100:.2f}±{lf['std_f1']*100:.2f}" if lf else "n/a"
            og_f = f"{og['mean_f1']*100:.2f}±{og['std_f1']*100:.2f}" if og else "n/a"
            lines.append(f"| {fold} | {FOLD_BEARING[fold]} | {arm} | {lf_r} | {og_r} | {lf_f} | {og_f} |\n")
    lines.append("\n")

    lines.append("## 逐折 dual − single 方向与漂移（recall，pp）\n\n")
    lines.append("| fold | 变体 | leakfree Δ(dual-single) | 原协议 Δ(dual-single) | 方向一致？ | 漂移 \\|Δleakfree-Δ原协议\\| (pp) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for fold in FOLDS:
        for variant, single_arm, dual_arm in (("nokin", "single_nokin", "dual_nokin"),
                                                ("kin", "single_kin", "dual_kin")):
            lf_s = data[(single_arm, fold)]["leakfree"]
            lf_d = data[(dual_arm, fold)]["leakfree"]
            og_s = data[(single_arm, fold)]["original"]
            og_d = data[(dual_arm, fold)]["original"]
            if lf_s is None or lf_d is None or og_s is None or og_d is None:
                lines.append(f"| {fold} | {variant} | n/a | n/a | n/a | n/a |\n")
                continue
            delta_lf = (lf_d["mean_recall"] - lf_s["mean_recall"]) * 100
            delta_og = (og_d["mean_recall"] - og_s["mean_recall"]) * 100
            same_dir = (delta_lf >= 0) == (delta_og >= 0)
            drift = abs(delta_lf - delta_og)
            lines.append(f"| {fold} | {variant} | {delta_lf:+.2f} | {delta_og:+.2f} | "
                         f"{'是' if same_dir else '**否**'} | {drift:.2f} |\n")
    lines.append("\n")

    # overall (arm-level summary.json) context, if present
    lines.append("## 附：整体（跨 4 折）summary.json 对照\n\n")
    lines.append("| arm | leakfree mean_macro_recall | 原协议 mean_macro_recall |\n|---|---|---|\n")
    for arm, orig_name in ARM_MAP.items():
        lf_summary = leakfree_root / arm / "summary.json"
        og_summary = PROJECT_ROOT / "results" / orig_name / "summary.json"
        lf_val = "n/a"
        og_val = "n/a"
        if lf_summary.exists():
            d = json.loads(lf_summary.read_text())
            lf_val = f"{d['mean_macro_recall']*100:.2f}±{d['std_macro_recall']*100:.2f}"
        if og_summary.exists():
            d = json.loads(og_summary.read_text())
            og_val = f"{d['mean_macro_recall']*100:.2f}±{d['std_macro_recall']*100:.2f}"
        lines.append(f"| {arm} | {lf_val} | {og_val} |\n")
    lines.append("\n")
    lines.append("不下结论，判据留人工裁决（P3）。\n")

    out_path = Path(args.out)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
