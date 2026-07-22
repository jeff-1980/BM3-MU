#!/usr/bin/env python3
"""
Step 5 / T9-g / S6 (2026-07-21, text-surgery day 2): zero-training forward-only
latency microbenchmark for BM3 (SISO), BM2, 1D-CNN, Transformer-1D, single
batch, win_len=2048, on the project GPU (RTX A5000 laptop, sm_86).

Not a training run: instantiates each model with default hyperparameters
(n_sensors=1, n_classes=4, d_model=64), feeds one batch of random input at
batch_size=1 and batch_size=64, times 50 forward passes after 10 warmup
iterations (torch.cuda.synchronize() before/after each timed call), reports
mean +/- std latency in milliseconds. Supports the paper's linear-time
fusion claim with one measured data point; not a systematic scaling study.
"""
import time
import numpy as np
import torch

from bearmamba3.model import BearMamba3
from baselines.mamba2 import BearMamba2
from baselines.cnn1d import BearCNN1D
from baselines.transformer1d import BearTransformer1D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN_LEN = 2048
N_SENSORS = 1
N_CLASSES = 4
D_MODEL = 64
N_WARMUP = 10
N_TIMED = 50


def build_models():
    models = {}
    models["BearMamba-3 (SISO)"] = BearMamba3(
        d_model=D_MODEL, d_state=128, n_layers=4, n_sensors=N_SENSORS,
        n_classes=N_CLASSES, conv_stride=2, is_mimo=False, mimo_rank=4,
        dtype=torch.bfloat16,
    )
    models["BearMamba-3 (MIMO r=4, fwd-only)"] = BearMamba3(
        d_model=D_MODEL, d_state=128, n_layers=4, n_sensors=N_SENSORS,
        n_classes=N_CLASSES, conv_stride=2, is_mimo=True, mimo_rank=4,
        dtype=torch.bfloat16,
    )
    models["BearMamba-2"] = BearMamba2(
        d_model=D_MODEL, d_state=128, n_layers=4, n_sensors=N_SENSORS,
        n_classes=N_CLASSES, conv_stride=2,
    )
    models["1D-CNN"] = BearCNN1D(
        d_model=D_MODEL, n_layers=4, n_sensors=N_SENSORS, n_classes=N_CLASSES,
        conv_stride=2,
    )
    models["Transformer-1D"] = BearTransformer1D(
        d_model=D_MODEL, n_layers=4, n_sensors=N_SENSORS, n_classes=N_CLASSES,
        conv_stride=2,
    )
    return models


def time_model(model, batch_size):
    model = model.to(DEVICE).eval()
    is_bm3 = isinstance(model, BearMamba3)
    x_dtype = torch.bfloat16 if is_bm3 else torch.float32
    x = torch.randn(batch_size, N_SENSORS, WIN_LEN, device=DEVICE, dtype=x_dtype)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(N_TIMED):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
    return float(np.mean(times)), float(np.std(times, ddof=1))


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    models = build_models()
    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        row = {"params": n_params}
        for bs in (1, 64):
            mean_ms, std_ms = time_model(model, bs)
            row[f"bs{bs}_mean_ms"] = mean_ms
            row[f"bs{bs}_std_ms"] = std_ms
            print(f"{name:20s} bs={bs:3d}  {mean_ms:7.3f} +/- {std_ms:6.3f} ms  "
                  f"(params={n_params})")
        results[name] = row
    return results


if __name__ == "__main__":
    main()
