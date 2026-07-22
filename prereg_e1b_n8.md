---
title: 预注册 — E1b n=8 扩展（第三轮评审 M1）
created: 2026-07-21
status: locked-before-run
---

# 预注册：E1b XJTU-SY cross-condition 六臂 n=8 扩展

## 动机

第三轮评审 M1：对 XJTU-SY Cond2→Cond3 cross-condition 三骨干融合增益结果
（single vs dual × {1D-CNN, BearMamba-2, BearMamba-3 CE-only}，现有 n=5,
seeds 0-4）补 seed 5/6/7，与既有 n=5 合并为 n=8 池，用更大样本压力测试原
n=5 下 "5/5 seeds concordant, exact Wilcoxon p=0.063" 的结论是否存活。

施加逻辑与 CWRU E6（旗舰格加 seed 至 n=8）**完全对称**：同一套骨干/配置/
超参数只把 `seeds` 字段从 `[0,1,2,3,4]` 扩为新增 `[5,6,7]`（不重跑旧
seed，只补新 seed），跑完后与既有 seed 0-4 结果合并计算。

## 六臂清单（本次新跑，各 3 seed，共 18 run）

| 臂 | backbone | n_sensors | 既有 config（seed 0-4，不动） | 新 config（seed 5-7） |
|---|---|---|---|---|
| single-CNN | cnn1d | 1 | `experiments/exp_mext_e13_1dcnn_xjtu_cross/config.yaml` | `experiments/exp_mext_e13_1dcnn_xjtu_cross/config_newseed.yaml` |
| dual-CNN | cnn1d | 2 | `experiments/exp_e1_dual_baselines/xjtu_cross_dual_cnn/config_cross_dual_cnn.yaml` | `.../config_cross_dual_cnn_newseed.yaml` |
| single-BM2 | mamba2 | 1 | `experiments/exp_e1_dual_baselines/xjtu_cross_single_bm2/config_cross_single_bm2.yaml` | `.../config_cross_single_bm2_newseed.yaml` |
| dual-BM2 | mamba2 | 2 | `experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/config_cross_dual_bm2.yaml` | `.../config_cross_dual_bm2_newseed.yaml` |
| single-BM3 (CE-only) | mamba3 | 1 | `experiments/exp_xjtu/config_cross_nokin.yaml` | `experiments/exp_xjtu/config_cross_nokin_newseed.yaml` |
| dual-BM3 (CE-only) | mamba3 | 2 | `experiments/exp_xjtu/config_cross_dual_nokin.yaml` | `experiments/exp_xjtu/config_cross_dual_nokin_newseed.yaml` |

新 config 与既有 config **逐字段一致，仅 `seeds`（改为 `[5, 6, 7]`）与
`results_dir`（加 `_newseed` 后缀，避免覆盖既有 seed 0-4 产物）不同。

## 统计裁决规则（锁死，跑之前定，不看数据后改）

1. 三组 dual−single 配对（按骨干分组：CNN、BM2、BM3-CEonly），每组把新
   seed 5/6/7 与既有 seed 0/1/2/3/4 的 macro-F1 合并为 n=8 池，逐 seed
   配对做 **exact 两侧 Wilcoxon signed-rank 检验**（与方法节
   `subsec:train` 的预声明检验方向政策一致：这三组比较均无方向性
   预注册，一律双侧）。
2. **如实报告**，不做取舍：
   - 若三组均维持 8/8 同向 concordant → 沿用现有"三骨干独立复现同方向"
     叙事，仅把 n=5 数字换成 n=8 数字、p 值换成 n=8 精确双侧 p 值。
   - **若任意一组从 5/5 同向掉到 8 个里出现反号（即失去 8/8 同向）**，
     该组在正文的措辞必须降级：不得再用"5/5 seeds concordant"或类似
     全同向表述，改为报告实际同向计数（如"7/8"）与对应 p 值，并如实
     注明"扩展后方向一致性减弱"。
3. 本规则与 CWRU E6 压力测试的处理逻辑对称：**扩大样本后若显著性/一致性
   蒸发，如实降级叙事，不回避、不换指标、不补种子到显著为止**。

## 落锁时间戳

本文件写入时间早于 E1b n=8 训练任务起跑（tmux session `e1b_n8`）。
