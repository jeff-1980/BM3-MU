#!/usr/bin/env python3
"""
experiments/exp_p2_cnn1d_ln/run_smoke_e2.py

一次性烟雾测试脚本（prereg_cnn_ln.md 任务 3）：对 cnn1d_ln 两个新配置
（CWRU clean / XJTU cross）各跑 2 epoch × 1 seed(=0) 的 --smoke 训练，验证
loss 下降、无 NaN，并汇总：
  - baselines/cnn1d.py vs baselines/cnn1d_ln.py 的逐键 diff 清单
  - cnn1d_ln 参数量（与 cnn1d 对比）
  - 全项目 grep "BatchNorm" 在 cnn1d_ln 前向路径（baselines/cnn1d_ln.py 本身）
    0 命中的证据

护栏：只调用既有 train.py 的 --smoke 路径（epoch<=2, step<=2 已由 train.py 内部
强制），本脚本自身不新增任何 >2 epoch 的训练路径。结果写入带时间戳的新目录
（results/e2_smoke_<ts>/），与满量队列前缀 results/e2_cnn_ln_* 严格隔离
（沿用 e1 阶段 reviewer F2/major 修复的隔离模式，见 run_smoke_e1.py）。

用法:
  cd <repo_root> && source venv/bin/activate
  python experiments/exp_p2_cnn1d_ln/run_smoke_e2.py
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
sys.path.insert(0, str(ROOT))

TS = "20260709-1524"
# ⚠️ 命名空间必须与 run_cnn_ln.sh 的满量输出前缀 (results/e2_cnn_ln_*) 完全隔离，
#   否则 status_e2.sh 的 glob (results/e2_cnn_ln_2*) 会把烟雾测试误判为满量运行产物。
SMOKE_DIR = ROOT / f"results/e2_smoke_{TS}/smoke"

CWRU_TRAIN = ROOT / "experiments/exp01_cwru_baseline/train.py"
XJTU_TRAIN = ROOT / "experiments/exp_xjtu/train.py"

ARMS = [
    {
        "key": "cwru_cnn1d_ln_clean",
        "base_cfg": ROOT / "experiments/exp_p2_cnn1d_ln/cwru/config_clean.yaml",
        "train": CWRU_TRAIN,
    },
    {
        "key": "xjtu_cross_cnn1d_ln",
        "base_cfg": ROOT / "experiments/exp_p2_cnn1d_ln/xjtu_cross/config.yaml",
        "train": XJTU_TRAIN,
    },
]


def make_smoke_config(arm: dict) -> Path:
    with open(arm["base_cfg"]) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
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


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def build_diff_and_param_evidence() -> dict:
    import torch
    from baselines.cnn1d import BearCNN1D
    from baselines.cnn1d_ln import BearCNN1D_LN

    diff = subprocess.run(
        ["diff", "-u", "baselines/cnn1d.py", "baselines/cnn1d_ln.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    ).stdout

    # Structural diff key-listing: only normalization-layer type differs.
    key_diff = [
        {"key": "class_name", "cnn1d": "BearCNN1D", "cnn1d_ln": "BearCNN1D_LN"},
        {"key": "conv_embed", "cnn1d": "Conv1d(k=8,stride=conv_stride,pad=3)",
         "cnn1d_ln": "Conv1d(k=8,stride=conv_stride,pad=3)  [unchanged]"},
        {"key": "norm_layer (x8, 2 per block x 4 blocks)",
         "cnn1d": "nn.BatchNorm1d(d_model)",
         "cnn1d_ln": "nn.GroupNorm(1, d_model)  [no batch statistics]"},
        {"key": "activation", "cnn1d": "GELU", "cnn1d_ln": "GELU  [unchanged]"},
        {"key": "pool", "cnn1d": "MaxPool1d(2)", "cnn1d_ln": "MaxPool1d(2)  [unchanged]"},
        {"key": "head", "cnn1d": "GlobalAvgPool + Linear",
         "cnn1d_ln": "GlobalAvgPool + Linear  [unchanged]"},
    ]

    cfg_common = dict(d_model=64, n_layers=4, n_sensors=1, n_classes=4, conv_stride=2)
    m_bn = BearCNN1D(**cfg_common)
    m_ln = BearCNN1D_LN(**cfg_common)
    n_bn = param_count(m_bn)
    n_ln = param_count(m_ln)

    # GroupNorm(1,C) has 2*C learnable params per layer (weight+bias), identical
    # shape to BatchNorm1d(C)'s learnable affine params; BatchNorm additionally
    # carries non-trainable running_mean/running_var buffers (not counted in
    # nn.Module.parameters()), so trainable param counts should match exactly.

    # Raw textual grep (task literal instruction: "grep BatchNorm").
    grep_bn = subprocess.run(
        ["grep", "-n", "BatchNorm", "baselines/cnn1d_ln.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    grep_bn_project = subprocess.run(
        ["grep", "-rn", "BatchNorm", "baselines/cnn1d_ln.py",
         "experiments/exp_p2_cnn1d_ln/"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    raw_hit_lines = grep_bn.stdout.splitlines()
    # Distinguish "mentioned in module docstring narrative" (harmless, describes
    # the ablation) from "appears in executable code" (would mean BN is still
    # instantiated). cnn1d_ln.py's module docstring (lines 1-19) documents the
    # replacement in prose and legitimately contains the word "BatchNorm1d".
    src_lines = (ROOT / "baselines/cnn1d_ln.py").read_text().splitlines()
    docstring_end = next(
        i for i, l in enumerate(src_lines) if i > 0 and l.strip() == '"""'
    ) + 1  # 1-indexed line number of closing triple-quote
    code_hits = [
        ln for ln in raw_hit_lines
        if int(ln.split(":", 1)[0]) > docstring_end
    ]

    # Definitive runtime proof: no nn.BatchNorm1d submodule is ever instantiated
    # in the ln arm's forward path, regardless of what any comment says.
    has_batchnorm_module = any(
        isinstance(m, torch.nn.BatchNorm1d) for m in m_ln.modules()
    )

    return {
        "diff_cnn1d_vs_cnn1d_ln": diff,
        "structural_key_diff": key_diff,
        "param_count": {
            "cnn1d (BatchNorm1d, trainable)": n_bn,
            "cnn1d_ln (GroupNorm, trainable)": n_ln,
            "delta": n_ln - n_bn,
        },
        "grep_batchnorm_in_cnn1d_ln_forward_path": {
            "cmd": "grep -n BatchNorm baselines/cnn1d_ln.py",
            "n_raw_hits": len(raw_hit_lines),
            "raw_output": grep_bn.stdout,
            "note": (
                f"docstring spans lines 1-{docstring_end}; the 1 raw hit is the "
                "module docstring's prose description of the ablation ('...仅将全部 "
                "8 处 nn.BatchNorm1d(d_model) 替换为 nn.GroupNorm...'), not "
                "executable code."
            ),
            "n_hits_in_executable_code (line > docstring_end)": len(code_hits),
            "hits_in_executable_code": code_hits,
        },
        "grep_batchnorm_in_ln_arm_files": {
            "cmd": "grep -rn BatchNorm baselines/cnn1d_ln.py experiments/exp_p2_cnn1d_ln/",
            "n_hits": len(grep_bn_project.stdout.splitlines()),
            "output": grep_bn_project.stdout,
        },
        "runtime_module_check": {
            "method": "any(isinstance(m, torch.nn.BatchNorm1d) for m in BearCNN1D_LN(...).modules())",
            "has_batchnorm1d_submodule": has_batchnorm_module,
        },
    }


def main():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    smoke_results = [run_smoke(arm) for arm in ARMS]
    evidence = build_diff_and_param_evidence()

    unitcheck = {
        "task": "cnn1d_ln arm unit check (prereg_cnn_ln.md item 3)",
        "smoke_results": smoke_results,
        **evidence,
    }

    out_path = SMOKE_DIR.parent / "e2_unitcheck.json"
    with open(out_path, "w") as f:
        json.dump(unitcheck, f, indent=2, ensure_ascii=False)
    out_path.chmod(0o444)
    print(f"\n{'='*70}\ne2_unitcheck.json written to {out_path} (chmod 444)")

    for r in smoke_results:
        status = "PASS" if r.get("passed") and r.get("loss_decreased") and not r.get("has_nan") else "FAIL"
        print(f"  [{status}] {r['arm']}  loss_decreased={r.get('loss_decreased')}  "
              f"has_nan={r.get('has_nan')}")
    print(f"  param_count: {evidence['param_count']}")
    ev_bn = evidence["grep_batchnorm_in_cnn1d_ln_forward_path"]
    print(f"  grep BatchNorm raw hits in cnn1d_ln.py: {ev_bn['n_raw_hits']} "
          f"(executable-code hits: {ev_bn['n_hits_in_executable_code (line > docstring_end)']})")
    print(f"  runtime check — has BatchNorm1d submodule: "
          f"{evidence['runtime_module_check']['has_batchnorm1d_submodule']}")

    all_pass = all(
        r.get("passed") and r.get("loss_decreased") and not r.get("has_nan")
        for r in smoke_results
    )
    grep_clean = ev_bn["n_hits_in_executable_code (line > docstring_end)"] == 0
    runtime_clean = not evidence["runtime_module_check"]["has_batchnorm1d_submodule"]
    sys.exit(0 if (all_pass and grep_clean and runtime_clean) else 1)


if __name__ == "__main__":
    main()
