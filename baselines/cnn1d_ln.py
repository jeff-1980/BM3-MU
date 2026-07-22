"""
baselines/cnn1d_ln.py — 1D-CNN 逐样本归一化（GroupNorm）消融版

与 cnn1d.py 完全相同，仅将全部 8 处 nn.BatchNorm1d(d_model) 替换为
nn.GroupNorm(1, d_model)（num_groups=1，对每个样本的全部通道联合归一化，
不使用任何 batch 统计，等价于 conv 场景下的 LayerNorm）。

用途：与 baselines/onedcnn_nobn.py（去 BN）互补，隔离"批统计"这一具体机制
是否是 1D-CNN 在低 SNR 场景优势的根因，而非"有无归一化"本身
（M_ext Phase 2b BLOCK-P2-1 后续追加消融，prereg_cnn_ln.md 记录）。

结构（与 cnn1d.py 逐键一致，仅归一化层类型不同）：
  - conv_embed：Conv1d, kernel=8, stride=conv_stride, padding=3（无 BN/GELU）
  - 4 个双卷积块：Conv1d → GroupNorm(1,C) → GELU → Conv1d → GroupNorm(1,C) →
    GELU → MaxPool1d(2)
  - 全局平均池化 → 线性分类头
  - 不支持 return_kin（纯 CE 基线，无物理先验）
"""
import torch.nn as nn


class BearCNN1D_LN(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_sensors: int = 1, n_classes: int = 4,
                 conv_stride: int = 2, **kwargs):
        super().__init__()
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(1, d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(1, d_model),
                nn.GELU(),
                nn.MaxPool1d(2),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin: bool = False):
        x = self.conv_embed(x)          # (B, d, L/stride)
        for blk in self.blocks:
            x = blk(x)                  # (B, d, L/2^n_layers)
        x = x.mean(-1)                  # (B, d) global avg pool
        logits = self.classifier(x)
        if return_kin:
            return logits, None
        return logits
