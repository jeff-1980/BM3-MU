#!/bin/bash
# experiments/exp_p2_cnn1d_ln/status_e2.sh
#
# 查看 run_cnn_ln.sh 排队任务的进度，不 attach tmux。
# ⚠️ glob 只匹配 results/e2_cnn_ln_2*（即 run_cnn_ln.sh 满量队列的时间戳输出
#   目录），不匹配项目里其它任何既有 results/** 目录，避免误报无关实验的完成
#   状态。烟雾测试/单元核查产物专用 results/e2_smoke_* 前缀（run_smoke_e2.py），
#   与本 glob 严格隔离，不会被误判为满量运行（沿用 e1 阶段 reviewer F2/major
#   修复的隔离模式）。
#
# 用法:
#   bash experiments/exp_p2_cnn1d_ln/status_e2.sh [ts_dir_name]
#   # 不传参数时列出所有匹配的时间戳目录并汇总每个目录内的完成情况

cd "$(dirname "$0")/../.." || exit 1

EXPECTED_SUBRUNS=7   # 1 arm × 6 SNR点(含clean) (CWRU cnn1d_ln) + 1 (XJTU cross cnn1d_ln)

shopt -s nullglob
DIRS=(results/e2_cnn_ln_2*)
shopt -u nullglob

if [ ${#DIRS[@]} -eq 0 ]; then
  echo "No results/e2_cnn_ln_2* directories found yet."
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
