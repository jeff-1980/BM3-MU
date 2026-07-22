"""
experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py

XJTU-SY Cond3 LOBO — 无泄漏选点协议（E3，见 prereg_lobo_leakfree.md）。

与 experiments/exp_xjtu/train.py 的关系：
  这是一个独立新文件，不修改、不 import 原 train.py 的 train_one_run()/run_lobo()。
  原脚本的既有 LOBO 训练/选点路径（每 epoch 用测试折指标取 best_macro_recall）保持不动，
  其 results/exp_xjtu_lobo_* 产物不受本文件影响。

本文件落地 prereg_lobo_leakfree.md 的三条规则：
  P1 fixed_epoch 选点：最终指标只取训练循环结束后最后一个 epoch，零选择。
  P2 测试折全程不可见：eval_recall_f1(model, test_loader, ...) 在整个脚本中只出现一次
     调用点，且位于训练 for-epoch 循环之外。
  P3 checkpoint 落盘：训练完成后可选保存模型权重，一次性投入、长期复用。

用法:
  source venv/bin/activate
  python experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py \
      --config experiments/exp_e3_lobo_leakfree/config_lobo_nokin_leakfree.yaml --smoke
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_xjtu import XJTUDataset, make_lobo_folds, BEARING_FAILURE
from bearmamba3.kinematic_loss import kinematic_loss
from bearmamba3.model import BearMamba3

# Cond3 LOBO bearings — identical set to experiments/exp_xjtu/train.py (D26), not redefined loosely.
COND3_LOBO_BEARINGS = ['Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "results_dir"):
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    return BearMamba3(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        n_sensors=cfg.get("n_sensors", 1),
        n_classes=cfg.get("n_classes", 2),
        conv_stride=cfg["conv_stride"],
        is_mimo=False,
        use_batchnorm=cfg.get("use_batchnorm", False),
        dtype=torch.bfloat16,
    ).to(device)


def make_train_loader(dataset: XJTUDataset, batch_size: int, num_workers: int, smoke: bool):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0 if smoke else num_workers,
        pin_memory=True, drop_last=True,
    )


def eval_recall_f1(model, loader, device, n_classes: int = 2):
    """Return per-class recall, macro recall, macro F1."""
    model.eval()
    tp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    with torch.no_grad():
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(1).cpu().numpy()
            labs = labels.cpu().numpy()
            for c in range(n_classes):
                tp[c] += ((preds == c) & (labs == c)).sum()
                fn[c] += ((preds != c) & (labs == c)).sum()
                fp[c] += ((preds == c) & (labs != c)).sum()
    recall = tp / np.maximum(tp + fn, 1).astype(float)
    precision = tp / np.maximum(tp + fp, 1).astype(float)
    f1_per = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    macro_recall = recall.mean()
    macro_f1 = f1_per.mean()
    return recall, macro_recall, macro_f1


# ── Per-fold / per-seed training (P1 + P2 + P3) ────────────────────────────────

def train_one_run_leakfree(
    cfg: dict,
    seed: int,
    train_ds: XJTUDataset,
    test_ds: XJTUDataset,
    device: torch.device,
    smoke: bool,
    fold_tag: str,
) -> dict:
    set_seed(seed)

    selection_mode = cfg.get("selection_mode", "fixed_epoch")
    assert selection_mode == "fixed_epoch", (
        f"P1 violation: selection_mode={selection_mode!r} not supported — "
        f"this script only implements the zero-selection fixed_epoch control (prereg P1)."
    )

    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)
    lambda_kin = cfg.get("lambda_kin", 0.0)
    kin_variant = cfg.get("kin_variant", "cover")
    bkw = cfg.get("bearing_kwargs", {})
    fs_eff = float(cfg["fs_eff"])
    n_epochs = 2 if smoke else cfg["epochs"]
    results_dir = cfg["results_dir"]
    do_kin = lambda_kin > 0

    train_loader = make_train_loader(train_ds, bs, nw, smoke)
    # P2: test_loader is constructed here but MUST NOT be iterated until after the
    # training loop below has fully completed. Grep evidence: the only call to
    # eval_recall_f1(model, test_loader, ...) in this file is after the `for epoch`
    # loop, outside its indentation block.
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        num_workers=0 if smoke else nw, pin_memory=True,
    )

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    t0 = time.time()

    # ── P2: TEST-FOLD-INVISIBLE TRAINING LOOP ──────────────────────────────
    # No reference to test_loader / test_ds anywhere in this block.
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_ce = running_kin = n_steps = 0

        for step, (x, labels, rpm) in enumerate(train_loader):
            if smoke and step >= 2:
                break
            x, labels, rpm = x.to(device), labels.to(device), rpm.to(device)

            if do_kin:
                out, kin = model(x, return_kin=True)
                l_ce = nn.functional.cross_entropy(out, labels)
                l_kin = kinematic_loss(kin, rpm, fs_eff,
                                        variant=kin_variant, bearing_kwargs=bkw)
                loss = l_ce + lambda_kin * l_kin
            else:
                out = model(x)
                l_ce = nn.functional.cross_entropy(out, labels)
                l_kin = torch.tensor(0.0)
                loss = l_ce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()

            running_ce += l_ce.item()
            running_kin += l_kin.item()
            n_steps += 1

        scheduler.step()

        # P1: no best-so-far tracking here. No comparison against any test-set metric.
        if epoch % 10 == 0 or epoch <= 3 or smoke:
            elapsed = time.time() - t0
            print(
                f"  [P2 test-fold-invisible] {fold_tag} seed={seed} ep={epoch:3d}/{n_epochs}"
                f"  l_ce={running_ce/max(n_steps,1):.4f}"
                f"  l_kin={running_kin/max(n_steps,1):.4f}"
                f"  [{elapsed:.0f}s]"
            )
    # ── end of test-fold-invisible training loop ───────────────────────────

    # P1 fixed_epoch selection: the ONLY test-fold evaluation in this run, taken
    # unconditionally at the final epoch (n_epochs). Zero selection.
    final_epoch = n_epochs
    recall, macro_recall, macro_f1 = eval_recall_f1(model, test_loader, device)
    print(
        f"  [P1 fixed_epoch] {fold_tag} seed={seed} final_epoch={final_epoch}"
        f"  recall=[{recall[0]:.3f},{recall[1]:.3f}]  macro_recall={macro_recall:.4f}"
        f"  macro_f1={macro_f1:.4f}  (first & only test-loader touch)"
    )

    # P3: checkpoint the final-epoch weights (one-time infra investment).
    checkpoint_path = None
    if cfg.get("save_checkpoint", False):
        ckpt_dir = Path(results_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"{fold_tag}_seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": final_epoch,
            "fold_tag": fold_tag,
            "seed": seed,
            "config_name": cfg.get("name"),
        }, checkpoint_path)

    return {
        "fold_tag": fold_tag,
        "seed": seed,
        "selection_mode": "fixed_epoch",
        "final_epoch": final_epoch,
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_recall": recall.tolist(),
        "elapsed_s": time.time() - t0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


# ── LOBO mode ─────────────────────────────────────────────────────────────────

def run_lobo_leakfree(cfg: dict, device: torch.device, smoke: bool,
                       max_folds: int = None, max_seeds: int = None):
    data_root = cfg["data_root"]
    seeds = cfg["seeds"] if max_seeds is None else cfg["seeds"][:max_seeds]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    folds = make_lobo_folds(COND3_LOBO_BEARINGS)  # 4 folds
    if max_folds is not None:
        folds = folds[:max_folds]

    all_seed_macro = []

    for seed in seeds:
        seed_fold_results = []
        or_recalls, ir_recalls = [], []

        for fold_idx, (train_bearings, test_bearings) in enumerate(folds):
            test_bearing = test_bearings[0]
            test_label = BEARING_FAILURE[test_bearing]
            fold_tag = f"fold{fold_idx}_{test_bearing}"

            out_path = results_dir / f"fold{fold_idx}_seed{seed}.json"
            if out_path.exists():
                print(f"  [SKIP] {out_path.name}")
                with open(out_path) as f:
                    res = json.load(f)
                seed_fold_results.append(res)
                recall_val = res["per_class_recall"][0 if test_label == 'OR' else 1]
                (or_recalls if test_label == 'OR' else ir_recalls).append(recall_val)
                continue

            print(f"\n{'='*60}")
            print(f"[E3 leak-free] Fold {fold_idx}: train={train_bearings} / test={test_bearing}({test_label})")

            n_sensors = cfg.get("n_sensors", 1)
            train_ds = XJTUDataset(data_root, train_bearings, n_sensors=n_sensors)
            test_ds = XJTUDataset(data_root, test_bearings, n_sensors=n_sensors)
            if smoke:
                train_ds._windows = train_ds._windows[:128]
                train_ds._labels = train_ds._labels[:128]
                train_ds._rpms = train_ds._rpms[:128]
                test_ds._windows = test_ds._windows[:64]
                test_ds._labels = test_ds._labels[:64]
                test_ds._rpms = test_ds._rpms[:64]

            res = train_one_run_leakfree(cfg, seed, train_ds, test_ds, device, smoke, fold_tag)
            seed_fold_results.append(res)

            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  -> saved {out_path}")

            recall_val = res["per_class_recall"][0 if test_label == 'OR' else 1]
            (or_recalls if test_label == 'OR' else ir_recalls).append(recall_val)

        if or_recalls and ir_recalls:
            macro = (np.mean(or_recalls) + np.mean(ir_recalls)) / 2
        elif or_recalls:
            macro = np.mean(or_recalls)
        elif ir_recalls:
            macro = np.mean(ir_recalls)
        else:
            macro = float("nan")

        all_seed_macro.append(float(macro))
        print(f"\n[E3] Seed {seed}: OR_recall={or_recalls}  IR_recall={ir_recalls}  macro={macro:.4f}")

    mean_macro = float(np.mean(all_seed_macro))
    std_macro = float(np.std(all_seed_macro, ddof=1)) if len(all_seed_macro) > 1 else 0.0
    summary = {
        "mode": "lobo_leakfree",
        "selection_mode": "fixed_epoch",
        "config": cfg.get("name"),
        "lambda_kin": cfg.get("lambda_kin", 0.0),
        "n_sensors": cfg.get("n_sensors", 1),
        "lobo_bearings": [te[0] for _, te in folds],
        "seeds": seeds,
        "per_seed_macro_recall": all_seed_macro,
        "mean_macro_recall": mean_macro,
        "std_macro_recall": std_macro,
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"[E3 leak-free] LOBO RESULTS: macro_recall = {mean_macro*100:.2f}+/-{std_macro*100:.2f}%")
    print(f"Per-seed: {[f'{v*100:.2f}' for v in all_seed_macro]}")
    print(f"Summary saved to {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--max-folds", type=int, default=None,
                     help="restrict number of LOBO folds run (unit-check use only)")
    ap.add_argument("--results-dir", default=None,
                     help="override cfg['results_dir'] (avoids editing/hardcoding "
                          "per-invocation paths into the yaml; e.g. a fresh "
                          "timestamped batch dir supplied by run_lobo_leakfree.sh)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cwd_rel = Path.cwd() / args.config
        cfg_path = cwd_rel if cwd_rel.exists() else Path(__file__).parent / args.config
    cfg = load_config(cfg_path)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).expanduser())
    if args.seeds:
        cfg["seeds"] = args.seeds

    assert "fs_eff" in cfg, "config.yaml must contain fs_eff (expected 12800 for XJTU)"
    assert abs(cfg["fs_eff"] - 12800.0) < 1, \
        f"fs_eff={cfg['fs_eff']} — expected 12800 (fs=25600 / conv_stride=2)"
    assert cfg.get("selection_mode", "fixed_epoch") == "fixed_epoch", \
        "P1 violation: this script only supports selection_mode=fixed_epoch"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E3 leak-free] Device: {device}  selection_mode=fixed_epoch"
          f"  n_sensors={cfg.get('n_sensors',1)}  lambda_kin={cfg.get('lambda_kin',0)}"
          f"  fs_eff={cfg['fs_eff']}  seeds={cfg['seeds']}"
          f"  epochs={2 if args.smoke else cfg['epochs']}"
          f"  save_checkpoint={cfg.get('save_checkpoint', False)}")

    run_lobo_leakfree(cfg, device, args.smoke, max_folds=args.max_folds)


if __name__ == "__main__":
    main()
