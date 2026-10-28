# c-infer

A pure C11 neural network inference engine built from scratch. No BLAS, no
external matrix libraries, no dynamic memory during the forward pass.

Trained models are exported from PyTorch via `exporter/export.py` into a
custom `.cinf` binary format and executed by the C engine in a single
allocation-free dispatch loop.

See `SPECIFICATION.md` for the original design spec.

## Layout

```
  include/cinfer.h         Public C API
  src/                     Engine, arena, tensor, loader
    layers/                Linear, Conv2D, MaxPool2D, Flatten
    activations/           ReLU, Sigmoid, Softmax
  exporter/export.py       PyTorch -> .cinf
  tests/                   Per-layer parity tests vs PyTorch
  demos/                   End-to-end MLP demo
  Makefile                 lib / test / demo / all / clean
```

## Parity with PyTorch

Each layer and a full MLP are exported from PyTorch with fixed weights and
a fixed input. The C engine output is compared against PyTorch's output.
`make test` runs all of them.

| Test | Covers | Tolerance vs PyTorch (abs.) |
|------|--------|-------------------------------|
| `test_linear` | fully connected | 1e-5 |
| `test_conv2d` | 2D convolution | 1e-5 |
| `test_maxpool2d` | max pooling | bit-exact |
| `test_flatten` | reshape | bit-exact |
| `test_relu` | ReLU | 1e-6 |
| `test_sigmoid` | sigmoid | 1e-6 |
| `test_softmax` | softmax, sums to 1 | 1e-6 (sum 1e-5) |
| `test_end_to_end` | MLP, softmax output | 1e-5 |
| `test_no_alloc_forward` | `malloc`/`calloc`/`realloc`/`free` wrapped at link time, forward pass must not call them | 0 allocations in forward |

Float32 throughout, built with `-ffp-contract=off`.

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

## Targeting bare-metal / MCU

The engine has no OS dependencies in the inference path. `cinf_forward`
does no allocation and no I/O; it walks a static array of
ops over buffers sized at load time.

Current status, stated plainly:

- `cinf_load_model` uses libc allocation (a temporary layer table and
  the arena). On a target without a heap, replace them with a static buffer.
- Weights are copied into the arena. Executing them in place from flash is
  not implemented.
- Only float32 is supported. No quantization yet.
- Tested on desktop Linux only. An MCU port is not done.

## Architecture

- **Bump arena** (`src/arena.c`): a single contiguous block for all
  weights + per-layer activation slots. Frozen after load.
- **Static execution graph**: `.cinf` encodes a pre-topologically-sorted
  array of op nodes. `cinf_forward` is one `for` loop.
- **Row-major NCHW** tensors; rank ∈ {1, 2, 4}.
- **`.cinf` v1 format**: little-endian, version-tagged, self-describing.

See `SPECIFICATION.md` for the full design contract.
