#!/usr/bin/env python3
"""
experiments/exp_p2_cnn1d_ln/run_one.py

小工具：加载一个既有 config.yaml，覆盖 results_dir（和可选 name），写临时 yaml，
再调用对应 train.py。供 run_cnn_ln.sh 把每条 (arm, SNR) 的产出统一收纳到一个
带时间戳的父目录 results/e2_cnn_ln_<ts>/ 下，而不改动
experiments/exp_p2_cnn1d_ln/**/config_*.yaml 里写死的规范 results_dir
（那些是给"单独复跑这一条"用的规范路径，e2_cnn_ln_<ts>/ 是这次批量排队运行的
额外聚合视图，只增不覆盖）。

逐字复用 experiments/exp_e1_dual_baselines/run_one.py 的实现（不改动那份文件，
本文件是独立副本，供本轮任务自包含）。

幂等：目标 results_dir/summary.json 已存在则跳过（--smoke 模式不检查，总是跑）。

用法:
  python experiments/exp_p2_cnn1d_ln/run_one.py \
      --train experiments/exp01_cwru_baseline/train.py \
      --config experiments/exp_p2_cnn1d_ln/cwru/config_snr0.yaml \
      --results-dir results/e2_cnn_ln_<ts>/cwru_snr0 \
      [--smoke] [--seeds 0 1 2 3 4]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["results_dir"] = str(Path(args.results_dir).expanduser())
    if "name" in cfg:
        cfg["name"] = cfg["name"] + "_e2batch"

    results_dir = Path(cfg["results_dir"])
    summary_path = results_dir / "summary.json"
    if summary_path.exists() and not args.smoke:
        print(f"  [SKIP] {results_dir} — summary.json exists")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_cfg = f.name

    cmd = [sys.executable, args.train, "--config", tmp_cfg]
    if args.smoke:
        cmd += ["--smoke"]
    if args.seeds:
        cmd += ["--seeds"] + [str(s) for s in args.seeds]

    print(f"  [RUN ] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
