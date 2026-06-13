"""export.py -- PyTorch -> .cinf exporter for the c-infer engine.

Supports the MVP layer/activation set:
    nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.Flatten,
    nn.ReLU, nn.Sigmoid, nn.Softmax(dim=...)

The exporter walks the model in declaration order and emits a single
.cinf blob. On disk:
    4 bytes  magic  "CINF"
    4 bytes  version 1   (little-endian uint32)
    4 bytes  n_layers    (LE uint32)
    4 bytes  weight_bytes (LE uint32, advisory, re-derived on load)
    --- per layer ---
    4 bytes  op_kind      (LE uint32, see cinf_op_kind_t in cinfer.h)
    4 bytes  in_rank      (LE uint32)
    4 bytes  in_shape[4]  (LE uint32, unused dims = 1)
    4 bytes  out_rank
    4 bytes  out_shape[4]
    4 bytes  param_bytes  (LE uint32, multiple of 4)
    N bytes  param_payload (raw little-endian float32)
"""

from __future__ import annotations
import io
import struct
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn


SUPPORTED_LINEAR = (nn.Linear,)
SUPPORTED_CONV = (nn.Conv2d,)
SUPPORTED_POOL = (nn.MaxPool2d,)
SUPPORTED_FLATTEN = (nn.Flatten,)
SUPPORTED_RELU = (nn.ReLU,)
SUPPORTED_SIGMOID = (nn.Sigmoid,)
SUPPORTED_SOFTMAX = (nn.Softmax,)


class UnsupportedLayerError(Exception):
    """Raised when the exporter encounters a layer kind it cannot handle."""


CINF_VERSION = 1


def _f32_le_bytes(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr.astype(np.float32)).tobytes(order="C")


def _build_record(
    op_kind: int,
    in_shape: tuple,
    out_shape: tuple,
    param_payload: bytes = b"",
) -> bytes:
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise UnsupportedLayerError("internal: shapes must be length 4")
    buf = io.BytesIO()
    buf.write(struct.pack("<I", op_kind))
    buf.write(struct.pack("<I", 4))                 # in_rank (always 4 slots on disk)
    buf.write(struct.pack("<4I", *in_shape))
    buf.write(struct.pack("<I", 4))                 # out_rank
    buf.write(struct.pack("<4I", *out_shape))
    buf.write(struct.pack("<I", len(param_payload)))
    buf.write(param_payload)
    return buf.getvalue()


def _shape_of(t: torch.Tensor) -> tuple:
    if t.dim() == 1:
        return (t.shape[0], 1, 1, 1)
    if t.dim() == 2:
        return (t.shape[0], t.shape[1], 1, 1)
    if t.dim() == 4:
        return tuple(t.shape)
    raise UnsupportedLayerError(
        f"rank {t.dim()} tensors not supported (only 1, 2, 4)"
    )


def _as_input_shape(t: torch.Tensor) -> tuple:
    """Shape the C engine expects to see as the input to the first op."""
    if t.dim() == 1:
        return (1, t.shape[0], 1, 1)
    if t.dim() == 2:
        return (1, t.shape[1], 1, 1)
    if t.dim() == 4:
        return tuple(t.shape)
    raise UnsupportedLayerError(f"rank {t.dim()} not supported")


def _emit(model: nn.Module, sample: torch.Tensor) -> bytes:
    """Walk `model` against `sample` and return a .cinf blob."""
    model.eval()
    with torch.no_grad():
        x = sample
        records: list[bytes] = []
        cur = x
        prev_out_shape: tuple | None = None

        for module in model.modules():
            if module is model:
                # Skip the root (we handle its children below).
                continue
            in_shape = _as_input_shape(cur)
            if isinstance(module, SUPPORTED_LINEAR):
                # nn.Linear stores weight as (out_features, in_features)
                W = module.weight.detach().cpu().contiguous().numpy()
                b = (
                    module.bias.detach().cpu().contiguous().numpy()
                    if module.bias is not None
                    else np.zeros(W.shape[0], dtype=np.float32)
                )
                payload = _f32_le_bytes(W) + _f32_le_bytes(b)
                # Forward
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(1, in_shape, out_shape, payload))
                cur = y
            elif isinstance(module, SUPPORTED_CONV):
                # Conv2d weight: (out_c, in_c, k, k)
                W = module.weight.detach().cpu().contiguous().numpy()
                b = (
                    module.bias.detach().cpu().contiguous().numpy()
                    if module.bias is not None
                    else np.zeros(W.shape[0], dtype=np.float32)
                )
                payload = _f32_le_bytes(W) + _f32_le_bytes(b)
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(2, in_shape, out_shape, payload))
                cur = y
            elif isinstance(module, SUPPORTED_POOL):
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(3, in_shape, out_shape, b""))
                cur = y
            elif isinstance(module, SUPPORTED_FLATTEN):
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(4, in_shape, out_shape, b""))
                cur = y
            elif isinstance(module, SUPPORTED_RELU):
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(5, in_shape, out_shape, b""))
                cur = y
            elif isinstance(module, SUPPORTED_SIGMOID):
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(6, in_shape, out_shape, b""))
                cur = y
            elif isinstance(module, SUPPORTED_SOFTMAX):
                # We expect Softmax(dim=...) matching the engine's row.
                if module.dim is None or module.dim < 0:
                    # The C engine normalizes the entire tensor when rank != 4
                    # or along the last dim for rank 4. Match nn.Softmax(dim=-1).
                    pass
                y = module(cur)
                out_shape = _as_input_shape(y)
                records.append(_build_record(7, in_shape, out_shape, b""))
                cur = y
            else:
                raise UnsupportedLayerError(
                    f"layer of type {type(module).__name__} is not supported by c-infer"
                )

        # Header
        weight_bytes = sum(len(r) - 28 for r in records)  # payload-only
        # Each record has a 28-byte fixed prefix (op_kind + rank + 4 shape +
        # rank + 4 shape + param_bytes = 4*7 = 28).
        header = struct.pack("<4sIII", b"CINF", CINF_VERSION, len(records), weight_bytes)
        return header + b"".join(records)


def export_model(model: nn.Module, sample: torch.Tensor, out_path: str | bytes) -> None:
    """Export a supported `model` to a .cinf file at `out_path`.

    `sample` is a representative input tensor used to walk the model
    and infer shapes. Its values are not embedded in the .cinf file.
    """
    blob = _emit(model, sample)
    if isinstance(out_path, (bytes, bytearray)):
        # Allow callers to pass an open file-like or a path.
        if hasattr(out_path, "write"):
            out_path.write(blob)
            return
        out_path = bytes(out_path)
    with open(out_path, "wb") as f:
        f.write(blob)


def export_sequential(seq: nn.Sequential, sample: torch.Tensor, out_path: str | bytes) -> None:
    """Convenience wrapper for nn.Sequential models."""
    export_model(seq, sample, out_path)


__all__ = ["export_model", "export_sequential", "UnsupportedLayerError"]
