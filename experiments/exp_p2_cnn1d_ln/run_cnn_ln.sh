#!/bin/bash
# experiments/exp_p2_cnn1d_ln/run_cnn_ln.sh
#
# cnn1d_ln 臂满量训练队列（预注册见 prereg_cnn_ln.md，任务 4）
#
#   CWRU DE, SNR ∈ {-8,-6,-4,-2,0}dB + clean × 5 seeds:
#     - cnn1d_ln  (experiments/exp_p2_cnn1d_ln/cwru/config_{snr-8,...,clean}.yaml)
#   XJTU H, Cond2→Cond3 cross, 5 seeds:
#     - cnn1d_ln  (experiments/exp_p2_cnn1d_ln/xjtu_cross/config.yaml)
#
# 输出：results/e2_cnn_ln_<ts>/<arm>[_snr<tag>]/{seed_*.json,summary.json}
#       （聚合视图；不覆盖各 arm config 里写死的规范 results_dir，那些路径完全独立，
#        见 experiments/exp_p2_cnn1d_ln/{cwru,xjtu_cross}/config*.yaml 的 results_dir 字段）
# 幂等：按 results/e2_cnn_ln_<ts>/<arm...>/summary.json 是否存在跳过
#       （<ts> 是本次调用时刻的时间戳——同一次调用内幂等，不同次调用会开新目录；
#        如需真正跨调用幂等，运行前手动 export TS=<已存在的时间戳> 再执行本脚本）
#
# ⚠️ 本脚本本轮任务只写不跑（护栏 0：loop 内禁 >2 epoch / >2 cell 训练）。
#    烟雾测试证据见 results/e2_smoke_20260709-1524/e2_unitcheck.json
#    （由 run_smoke_e2.py 产生，命名空间与本脚本的满量队列前缀 results/e2_cnn_ln_*
#     严格隔离，二者互不复用输出目录）。
#
# ── tmux 用法（真正执行满量训练时）──────────────────────────────────
#   tmux new -s e2_cnn_ln
#   cd "$(dirname "$0")/../.." && source venv/bin/activate
#   bash experiments/exp_p2_cnn1d_ln/run_cnn_ln.sh
#   # detach: Ctrl-b d
#   # 重新 attach: tmux attach -t e2_cnn_ln
#   # 查看进度（不 attach）: bash experiments/exp_p2_cnn1d_ln/status_e2.sh
#
# 预计规模：1 arm × 6 SNR点(含clean) × 5 seeds (CWRU) + 1 arm × 5 seeds (XJTU cross)
#         = 30 + 5 = 35 runs

set -e
cd "$(dirname "$0")/../.."
source venv/bin/activate

TS="${TS:-$(date +%Y%m%d-%H%M)}"
OUT_ROOT="results/e2_cnn_ln_${TS}"
echo "=== e2 cnn1d_ln queue — TS=${TS}  OUT_ROOT=${OUT_ROOT} ==="

RUN_ONE="experiments/exp_p2_cnn1d_ln/run_one.py"
CWRU_TRAIN="experiments/exp01_cwru_baseline/train.py"
XJTU_TRAIN="experiments/exp_xjtu/train.py"
SEEDS="0 1 2 3 4"
SNR_GRID="clean 0 -2 -4 -6 -8"   # 逐字继承 BN 版网格 (exp07_baselines/config_cnn1d_*.yaml)

snr_tag() {
  # clean -> clean ; 0 -> 0 ; -8 -> m8  (与 exp_b2_dual_sensor/run_b2_snr_curve.py::snr_tag 命名一致)
  local snr="$1"
  if [ "$snr" = "clean" ]; then
    echo "clean"
  elif [ "${snr:0:1}" = "-" ]; then
    echo "m${snr#-}"
  else
    echo "${snr}"
  fi
}

# ── CWRU cnn1d_ln SNR grid ──────────────────────────────────────
for snr in $SNR_GRID; do
  tag=$(snr_tag "$snr")
  if [ "$snr" = "clean" ]; then
    cfg="experiments/exp_p2_cnn1d_ln/cwru/config_clean.yaml"
  else
    cfg="experiments/exp_p2_cnn1d_ln/cwru/config_snr${snr}.yaml"
  fi
  python "$RUN_ONE" \
    --train "$CWRU_TRAIN" \
    --config "$cfg" \
    --results-dir "${OUT_ROOT}/cwru_cnn1d_ln_snr${tag}" \
    --seeds $SEEDS
done

# ── XJTU cross cnn1d_ln ──────────────────────────────────────────
python "$RUN_ONE" \
  --train "$XJTU_TRAIN" \
  --config "experiments/exp_p2_cnn1d_ln/xjtu_cross/config.yaml" \
  --results-dir "${OUT_ROOT}/xjtu_cross_cnn1d_ln" \
  --seeds $SEEDS

echo "=== e2 cnn1d_ln queue DONE — outputs under ${OUT_ROOT} ==="
