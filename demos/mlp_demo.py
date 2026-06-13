"""mlp_demo.py -- end-to-end demo: train a small MLP, export to .cinf,
print PyTorch reference output, and (if invoked with --no-print) skip
printing so the Makefile demo target can drive it non-interactively.

By default prints the PyTorch reference output to stdout.
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "exporter"))
from export import export_sequential  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None, help="output .cinf path")
    ap.add_argument("--no-print", action="store_true",
                    help="skip PyTorch reference printout")
    args = ap.parse_args()

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 4),
        nn.Softmax(dim=-1),
    )
    model.eval()
    x = torch.randn(1, 8)
    with torch.no_grad():
        y = model(x).detach().cpu().contiguous().numpy().astype("<f4")
        x_np = x.detach().cpu().contiguous().numpy().astype("<f4")

    out_path = args.export or os.path.join(ROOT, "build", "mlp.cinf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_sequential(model, x, out_path)
    # Also dump the input so the C CLI uses exactly the same vector.
    in_path = os.path.join(os.path.dirname(out_path), "mlp.input.bin")
    with open(in_path, "wb") as f:
        f.write(x_np.tobytes(order="C"))
    print(f"exported {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"input    {in_path}")

    if not args.no_print:
        print(f"\nMLP demo (Linear {8}->6 -> ReLU -> Linear 6->4 -> Softmax)")
        print(f"input  ({x_np.shape}): {x_np.tolist()}")
        print(f"output ({y.shape}):    {y.tolist()}")
        print("sum:    ", float(y.sum()))


if __name__ == "__main__":
    main()
