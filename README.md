# c-infer

A pure C11 neural network inference engine built from scratch. No BLAS, no
external matrix libraries, no dynamic memory during the forward pass.

Trained models are exported from PyTorch via `exporter/export.py` into a
custom `.cinf` binary format and executed by the C engine in a single
allocation-free dispatch loop.

See `SPECIFICATION.md` for the original design spec.

## Layout

```
projects/c-infer/
  include/cinfer.h         Public C API
  src/                     Engine, arena, tensor, loader
    layers/                Linear, Conv2D, MaxPool2D, Flatten
    activations/           ReLU, Sigmoid, Softmax
  exporter/export.py       PyTorch -> .cinf
  tests/                   Per-layer parity tests vs PyTorch
  demos/                   End-to-end MLP demo
  Makefile                 lib / test / demo / all / clean
```

## Build

Requires a C11 compiler (gcc or clang) and Python 3 with `torch` and
`numpy` for the exporter and demo.

```bash
make lib      # static library libcinfer.a
make test     # build and run all parity tests
make demo     # train a small MLP, export, run inference in C
make all      # lib + test + demo
make clean
```

C flags: `-std=c11 -Wall -Wextra -Wpedantic -O2 -ffp-contract=off`.

## Public API

```c
#include "cinfer.h"

cinf_status_t cinf_load_model (const void* buffer, size_t size, cinf_model_t* out);
void          cinf_free_model (cinf_model_t* model);
cinf_status_t cinf_forward    (const cinf_model_t* model, const float* input, float* output);
const char*   cinf_strerror   (cinf_status_t);
```

`cinf_load_model` performs all allocations (weights, biases, activation
slots) up front via a bump arena. `cinf_forward` performs zero
allocations and is safe to call repeatedly with the same loaded model.

## Architecture

- **Bump arena** (`src/arena.c`): a single contiguous block for all
  weights + per-layer activation slots. Frozen after load.
- **Static execution graph**: `.cinf` encodes a pre-topologically-sorted
  array of op nodes. `cinf_forward` is one `for` loop.
- **Row-major NCHW** tensors; rank ∈ {1, 2, 4}.
- **`.cinf` v1 format**: little-endian, version-tagged, self-describing.

See `SPECIFICATION.md` for the full design contract.
