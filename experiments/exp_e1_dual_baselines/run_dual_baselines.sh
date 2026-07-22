#!/bin/bash
# experiments/exp_e1_dual_baselines/run_dual_baselines.sh
#
# e1 dual-baseline 扩展臂满量训练队列（预注册见 prereg_dual_baselines.md）
#
#   CWRU DE+FE, SNR ∈ {-8,-6,-4,-2,0}dB × 5 seeds:
#     - dual_bm2          (新臂, experiments/exp_e1_dual_baselines/cwru_dual_bm2/)
#     - dual_cnn          (无缺档，直接复用既有
#                           experiments/exp_mext_e14_1dcnn_cwru_dual/config_snr*.yaml，
#                           见 e1_config_audit.md 任务2核对结论)
#     - cnn1d_attnfusion  (新臂, experiments/exp_e1_dual_baselines/cwru_cnn1d_attnfusion/)
#   XJTU H+V, Cond2→Cond3 cross, 5 seeds:
#     - dual_bm2          (新臂, experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/)
#
# 输出：results/e1_dual_baselines_<ts>/<arm>[_snr<tag>]/{seed_*.json,summary.json}
#       （聚合视图；不覆盖各 arm config 里写死的规范 results_dir，那些路径完全独立）
# 幂等：按 results/e1_dual_baselines_<ts>/<arm...>/summary.json 是否存在跳过
#       （注意 <ts> 是本次调用时刻的时间戳——同一次调用内幂等，不同次调用会开新目录；
#        如需真正跨调用幂等，运行前手动 export TS=<已存在的时间戳> 再执行本脚本）
#
# ⚠️ 本脚本本轮任务只写不跑（护栏 0：loop 内禁 >2 epoch / >2 cell 训练）。
#    烟雾测试证据见 results/e1_smoke_20260709-0001/e1_unitcheck.json
#    （由 run_smoke_e1.py 产生，命名空间与本脚本的满量队列前缀 results/e1_dual_baselines_*
#     严格隔离，二者互不复用输出目录——见 reviewer F2/major 修复记录）。
#
# ── tmux 用法（真正执行满量训练时）──────────────────────────────────
#   tmux new -s e1_dual_baselines
#   cd "$(dirname "$0")/../.." && source venv/bin/activate
#   bash experiments/exp_e1_dual_baselines/run_dual_baselines.sh
#   # detach: Ctrl-b d
#   # 重新 attach: tmux attach -t e1_dual_baselines
#   # 查看进度（不 attach）: bash experiments/exp_e1_dual_baselines/status_e1.sh
#
# 预计规模：3 arms × 5 SNR × 5 seeds (CWRU) + 1 arm × 5 seeds (XJTU cross) = 80 runs
# （dual_cnn 只是复用既有已完成结果的产物路径分派，若已跑过会被幂等跳过，不重复计入 GPU 时间）

set -e
cd "$(dirname "$0")/../.."
source venv/bin/activate

TS="${TS:-$(date +%Y%m%d-%H%M)}"
OUT_ROOT="results/e1_dual_baselines_${TS}"
echo "=== e1 dual baselines queue — TS=${TS}  OUT_ROOT=${OUT_ROOT} ==="

RUN_ONE="experiments/exp_e1_dual_baselines/run_one.py"
CWRU_TRAIN="experiments/exp01_cwru_baseline/train.py"
XJTU_TRAIN="experiments/exp_xjtu/train.py"
SEEDS="0 1 2 3 4"
SNR_GRID="0 -2 -4 -6 -8"   # 逐字继承 dual-BM3 网格 (run_b2_snr_curve.py::SNR_GRID)

snr_tag() {
  # 0 -> 0 ; -8 -> m8  (与 exp_b2_dual_sensor/run_b2_snr_curve.py::snr_tag 命名一致)
  local snr="$1"
  if [ "${snr:0:1}" = "-" ]; then
    echo "m${snr#-}"
  else
    echo "${snr}"
  fi
}

# ── CWRU dual_bm2 ────────────────────────────────────────────────
for snr in $SNR_GRID; do
  tag=$(snr_tag "$snr")
  python "$RUN_ONE" \
    --train "$CWRU_TRAIN" \
    --config "experiments/exp_e1_dual_baselines/cwru_dual_bm2/config_snr${snr}.yaml" \
    --results-dir "${OUT_ROOT}/cwru_dual_bm2_snr${tag}" \
    --seeds $SEEDS
done

# ── CWRU dual_cnn（无缺档，复用既有 e14 config，见任务2核对）────────
for snr in $SNR_GRID; do
  tag=$(snr_tag "$snr")
  python "$RUN_ONE" \
    --train "$CWRU_TRAIN" \
    --config "experiments/exp_mext_e14_1dcnn_cwru_dual/config_snr${snr}.yaml" \
    --results-dir "${OUT_ROOT}/cwru_dual_cnn_snr${tag}" \
    --seeds $SEEDS
done

# ── CWRU cnn1d_attnfusion（新臂）─────────────────────────────────
for snr in $SNR_GRID; do
  tag=$(snr_tag "$snr")
  python "$RUN_ONE" \
    --train "$CWRU_TRAIN" \
    --config "experiments/exp_e1_dual_baselines/cwru_cnn1d_attnfusion/config_snr${snr}.yaml" \
    --results-dir "${OUT_ROOT}/cwru_cnn1d_attnfusion_snr${tag}" \
    --seeds $SEEDS
done

# ── XJTU cross dual_bm2（新臂）───────────────────────────────────
python "$RUN_ONE" \
  --train "$XJTU_TRAIN" \
  --config "experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/config_cross_dual_bm2.yaml" \
  --results-dir "${OUT_ROOT}/xjtu_cross_dual_bm2" \
  --seeds $SEEDS

echo "=== e1 dual baselines queue DONE — outputs under ${OUT_ROOT} ==="
