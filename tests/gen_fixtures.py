"""Test fixture generator for per-layer parity tests.

For each MVP op we construct a tiny PyTorch model with fixed seed=42,
export it via exporter/export.py, run inference in PyTorch to produce
the reference output, and write both the .cinf file and the expected
output binary (little-endian float32) into build/test_fixtures/.

Each test_X.c in tests/ loads the corresponding fixture and asserts
parity within 1e-5 (or 1e-6 for activations / exact for pool,flatten).
"""
from __future__ import annotations
import os
import struct
import sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "exporter"))

from export import export_sequential  # noqa: E402

OUT = os.path.join(ROOT, "build", "test_fixtures")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)


def write_pair(name: str, model: nn.Module, x: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        y = model(x).detach().cpu().contiguous().numpy().astype("<f4")
    cinf_path = os.path.join(OUT, f"{name}.cinf")
    exp_path  = os.path.join(OUT, f"{name}.expected.bin")
    in_path   = os.path.join(OUT, f"{name}.input.bin")
    export_sequential(model, x, cinf_path)
    with open(exp_path, "wb") as f:
        f.write(y.tobytes(order="C"))
    x_np = x.detach().cpu().contiguous().numpy().astype("<f4")
    with open(in_path, "wb") as f:
        f.write(x_np.tobytes(order="C"))
    print(f"  {name}: cinf={os.path.getsize(cinf_path)}B in={x_np.shape} expected={y.shape} {y.dtype}")


def main() -> None:
    # 1. Linear (dense): 8 -> 5
    write_pair(
        "linear",
        nn.Sequential(nn.Linear(8, 5)),
        torch.randn(1, 8),
    )
    # 2. Conv2D: in=1, out=3, k=3, H=W=8
    write_pair(
        "conv2d",
        nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)),
        torch.randn(1, 1, 8, 8),
    )
    # 3. MaxPool2D: 2x2 stride 2
    write_pair(
        "maxpool2d",
        nn.Sequential(nn.MaxPool2d(2)),
        torch.randn(1, 2, 8, 8),
    )
    # 4. Flatten
    write_pair(
        "flatten",
        nn.Sequential(nn.Flatten()),
        torch.randn(1, 2, 3, 4),
    )
    # 5. ReLU
    write_pair(
        "relu",
        nn.Sequential(nn.ReLU()),
        torch.randn(1, 4) - 0.5,
    )
    # 6. Sigmoid
    write_pair(
        "sigmoid",
        nn.Sequential(nn.Sigmoid()),
        torch.randn(1, 4),
    )
    # 7. Softmax (rank 1)
    write_pair(
        "softmax",
        nn.Sequential(nn.Softmax(dim=-1)),
        torch.randn(1, 5),
    )
    # 8. End-to-end MLP: 8 -> 6 (ReLU) -> 4 (Softmax)
    write_pair(
        "end_to_end",
        nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4), nn.Softmax(dim=-1)),
        torch.randn(1, 8),
    )
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main()
