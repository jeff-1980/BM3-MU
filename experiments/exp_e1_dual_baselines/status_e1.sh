#!/bin/bash
# experiments/exp_e1_dual_baselines/status_e1.sh
#
# 查看 run_dual_baselines.sh 排队任务的进度，不 attach tmux。
# ⚠️ glob 只匹配 results/e1_dual_baselines_2*（即 run_dual_baselines.sh 满量队列的
#   时间戳输出目录），不匹配项目里其它任何既有 results/** 目录，避免误报无关实验的
#   完成状态。烟雾测试/单元核查产物专用 results/e1_smoke_* 前缀（run_smoke_e1.py），
#   与本 glob 严格隔离，不会被误判为满量运行（见 reviewer F2/major 修复记录）。
#
# 用法:
#   bash experiments/exp_e1_dual_baselines/status_e1.sh [ts_dir_name]
#   # 不传参数时列出所有匹配的时间戳目录并汇总每个目录内的完成情况

cd "$(dirname "$0")/../.." || exit 1

EXPECTED_SUBRUNS=16   # 3 arms × 5 SNR (CWRU: dual_bm2/dual_cnn/cnn1d_attnfusion) + 1 (XJTU cross dual_bm2)
                       # 注：dual_cnn 分支复用既有 config，仍计入本次聚合视图的独立 summary.json

shopt -s nullglob
DIRS=(results/e1_dual_baselines_2*)
shopt -u nullglob

if [ ${#DIRS[@]} -eq 0 ]; then
  echo "No results/e1_dual_baselines_2* directories found yet."
  exit 0
fi

for d in "${DIRS[@]}"; do
  [ -d "$d" ] || continue
  echo "=== $d ==="
  total=0
  done_count=0
  for sub in "$d"/*/; do
    [ -d "$sub" ] || continue
    total=$((total + 1))
    if [ -f "${sub}summary.json" ]; then
      done_count=$((done_count + 1))
      status="DONE"
    else
      n_seeds=$(find "$sub" -maxdepth 1 -name 'seed_*.json' 2>/dev/null | wc -l)
      status="RUNNING (${n_seeds} seed jsons so far)"
    fi
    printf "  %-45s %s\n" "$(basename "$sub")" "$status"
  done
  echo "  -> ${done_count}/${total} sub-runs complete in this dir (expected total: ${EXPECTED_SUBRUNS})"
  echo
done
