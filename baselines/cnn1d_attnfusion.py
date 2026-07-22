"""
baselines/cnn1d_attnfusion.py — 1D-CNN + 通道注意力早融合（单变量消融臂，e1）

在 baselines/cnn1d.py::BearCNN1D 之前插入一个最小通道注意力早融合模块：
对 n_sensors 个输入传感器通道做 squeeze-excitation 式加权（逐通道全局均值
→ 2 层 MLP → sigmoid → 逐通道缩放），再送入与 BearCNN1D 完全相同的共享
conv_embed + 卷积块。

与 baselines/cnn1d.py 的 diff（单变量，仅此模块，逐键列出）：
  + __init__ 新增一行: self.channel_attn = ChannelAttention(n_sensors)
  + forward   新增一行: x = self.channel_attn(x)   （在 conv_embed 之前）
  其余（conv_embed / blocks / classifier 的构造与 forward 顺序）与 cnn1d.py 逐字相同。
"""
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-excitation style per-sensor-channel attention (early fusion)."""

    def __init__(self, n_sensors: int, reduction: int = 2):
        super().__init__()
        hidden = max(n_sensors // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(n_sensors, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_sensors),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, n_sensors, L)
        w = x.mean(dim=-1)              # (B, n_sensors) squeeze
        w = self.fc(w).unsqueeze(-1)    # (B, n_sensors, 1) excite
        return x * w


class BearCNN1DAttnFusion(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_sensors: int = 1, n_classes: int = 4,
                 conv_stride: int = 2, **kwargs):
        super().__init__()
        self.channel_attn = ChannelAttention(n_sensors)
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.MaxPool1d(2),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin: bool = False):
        x = self.channel_attn(x)        # early fusion: reweight sensor channels
        x = self.conv_embed(x)          # (B, d, L/stride)
        for blk in self.blocks:
            x = blk(x)                  # (B, d, L/2^n_layers)
        x = x.mean(-1)                  # (B, d) global avg pool
        logits = self.classifier(x)
        if return_kin:
            return logits, None
        return logits
