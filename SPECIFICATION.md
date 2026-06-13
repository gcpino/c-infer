# c-infer Specification

## Overview
**c-infer** is a lightweight, pure C11 neural network inference engine designed for embedded systems and edge devices. It bridges the gap between models trained in Python (PyTorch) and environments where predictability, memory constraints, and strict execution control are critical.

## 1. Core Principles
- **No Dynamic Memory During Inference:** The engine must never call `malloc` or `free` during the forward pass. All memory for weights, biases, and intermediate activations is calculated and allocated strictly during an initialization phase (often in a static arena).
- **Zero External Dependencies:** Built relying solely on standard C (and specifically bounded `<math.h>` and `<stdint.h>`). No BLAS, no external matrix libraries.
- **Deterministic Execution:** The execution graph is static, represented as a linear array of operations.

## 2. Components

### 2.1 The Inference Engine (C11)
- **Memory Arena:** A simple linear allocator used exclusively during model loading.
- **Tensors:** N-dimensional array structs with contiguous memory blocks.
- **Operations:** Highly optimized C functions for matrix multiplication, convolutions, and pooling. SIMD intrinsics may be optionally implemented later.
- **API:**
  - `cinf_load_model(const void* buffer, size_t size, cinf_arena_t* arena)`
  - `cinf_forward(cinf_model_t* model, const float* input, float* output)`

### 2.2 The Exporter (Python / PyTorch)
- A Python tool `export.py` that takes a trained `torch.nn.Module`.
- Extracts weights, biases, and the layer sequence.
- Serializes the exact network topology and weights into a custom binary format (`.cinf`).

## 3. Supported Operations (MVP)
- **Layers:** Linear (Dense), Conv1D, Conv2D, Flatten, MaxPool2D
- **Activations:** ReLU, Sigmoid, Softmax

## 4. Development Workflow & Testing
- **Validation:** Every layer implemented in C must have a corresponding test asserting that its output matches the PyTorch equivalent for a given random seed, within a strict floating-point tolerance (`1e-5`).
- **Specification-Driven:** Behavior is defined first by Python reference implementations, then mirrored exactly in C.
