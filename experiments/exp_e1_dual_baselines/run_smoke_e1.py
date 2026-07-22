#!/usr/bin/env python3
"""
experiments/exp_e1_dual_baselines/run_smoke_e1.py

一次性烟雾测试脚本（任务 4）：对三个新臂（dual_bm2 CWRU / dual_bm2 XJTU cross /
cnn1d_attnfusion CWRU）各跑 clean（无噪声）2 epoch × 1 seed(=0) 的 --smoke 训练，
验证 loss 下降、无 NaN、输出 shape 正确，并记录参数量。

护栏：只调用既有 train.py 的 --smoke 路径（epoch<=2, step<=2 已由 train.py 内部
强制），本脚本自身不新增任何 >2 epoch 的训练路径。结果写入带时间戳的新目录。

用法:
  cd <repo_root> && source venv/bin/activate
  python experiments/exp_e1_dual_baselines/run_smoke_e1.py
"""
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
TS = "20260709-0001"
# ⚠️ 命名空间必须与 run_dual_baselines.sh 的满量输出前缀 (results/e1_dual_baselines_*)
#   完全隔离，否则 status_e1.sh 的 glob 会把烟雾测试误判为满量运行产物（见 reviewer F2/major）。
SMOKE_DIR = ROOT / f"results/e1_smoke_{TS}/smoke"

CWRU_TRAIN = ROOT / "experiments/exp01_cwru_baseline/train.py"
XJTU_TRAIN = ROOT / "experiments/exp_xjtu/train.py"

ARMS = [
    {
        "key": "cwru_dual_bm2",
        "base_cfg": ROOT / "experiments/exp_e1_dual_baselines/cwru_dual_bm2/config_snr0.yaml",
        "train": CWRU_TRAIN,
        "strip_noise": True,
    },
    {
        "key": "xjtu_cross_dual_bm2",
        "base_cfg": ROOT / "experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/config_cross_dual_bm2.yaml",
        "train": XJTU_TRAIN,
        "strip_noise": False,  # XJTU config has no noise_snr_db field (raw signal, no synthetic noise)
    },
    {
        "key": "cwru_cnn1d_attnfusion",
        "base_cfg": ROOT / "experiments/exp_e1_dual_baselines/cwru_cnn1d_attnfusion/config_snr0.yaml",
        "train": CWRU_TRAIN,
        "strip_noise": True,
    },
]


def make_smoke_config(arm: dict) -> Path:
    with open(arm["base_cfg"]) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if arm["strip_noise"]:
        cfg.pop("noise_snr_db", None)  # clean = no noise injection
    cfg["name"] = cfg["name"] + "_smoke"
    results_dir = SMOKE_DIR / arm["key"]
    cfg["results_dir"] = str(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        return Path(f.name)


LCE_RE = re.compile(r"l_ce=(nan|-?\d+\.\d+)")


def run_smoke(arm: dict) -> dict:
    cfg_path = make_smoke_config(arm)
    cmd = [sys.executable, str(arm["train"]), "--config", str(cfg_path),
           "--smoke", "--seeds", "0"]
    print(f"\n{'='*70}\n[SMOKE] {arm['key']}\n  cmd: {' '.join(cmd)}\n{'='*70}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log = proc.stdout + "\n" + proc.stderr
    print(log[-4000:])
    ok = proc.returncode == 0
    result = {
        "arm": arm["key"],
        "cmd": " ".join(cmd),
        "config_used": str(cfg_path),
        "returncode": proc.returncode,
        "passed": ok,
    }
    if ok:
        # Both train.py variants print "l_ce=<float>" every epoch when --smoke is set
        # (exp01: unconditional smoke print; exp_xjtu: `epoch <= 3 or smoke` print).
        # Parse from stdout since exp_xjtu's per-seed JSON has no per-epoch history.
        raw_matches = LCE_RE.findall(log)
        has_nan = any(m == "nan" for m in raw_matches)
        losses = [float(m) for m in raw_matches if m != "nan"]
        result["n_lce_prints"] = len(losses)
        result["l_ce_trace"] = losses
        result["loss_decreased"] = (len(losses) >= 2 and losses[-1] < losses[0])
        result["has_nan"] = has_nan
        if not losses:
            result["passed"] = False
            result["error"] = "no l_ce= lines found in stdout"
    else:
        result["error"] = log[-2000:]
    return result


def main():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_smoke(arm) for arm in ARMS]
    out_path = SMOKE_DIR / "smoke_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}\nSmoke results written to {out_path}")
    for r in results:
        status = "PASS" if r.get("passed") and r.get("loss_decreased") and not r.get("has_nan") else "FAIL"
        print(f"  [{status}] {r['arm']}  loss_decreased={r.get('loss_decreased')}  "
              f"has_nan={r.get('has_nan')}  l_ce={r.get('l_ce_per_epoch')}")
    all_pass = all(
        r.get("passed") and r.get("loss_decreased") and not r.get("has_nan")
        for r in results
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
