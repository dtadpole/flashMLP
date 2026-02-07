# FlashMLP

A memory-efficient, high-performance Multi-Head MLP implementation using Triton kernels, inspired by the mhmoe architecture.

## Overview

FlashMLP implements a fused multi-head MLP operation:
```
Z = X @ W1
H = activation(Z)
O = H @ W2
```

### Key Features

- **Fused Operations**: Combines matrix multiplication and activation in a single kernel
- **Memory Efficient**: Processes intermediate dimension in chunks to reduce memory usage
- **Multi-Head Support**: Efficiently handles multiple heads in parallel
- **Multiple Activations**: Supports SiLU, Leaky ReLU, Sigmoid, and no activation
- **Optimized Backward Pass**: Unified kernel for computing gradients
- **PyTorch Integration**: Full autograd support via custom Function

## Architecture

### Input/Output Shapes

- **X**: `(H, B, D)` - H heads, B batch size, D hidden dimension
- **W1**: `(H, D, E)` - First weight matrix (projects D → E)
- **W2**: `(H, E, D)` - Second weight matrix (projects E → D)
- **Output**: `(H, B, D)` - Same shape as input

### Implementation Details

#### Forward Kernel (`flashmlp_fwd_kernel`)

The forward kernel processes the MLP in a tiled manner:

1. **Tiling Strategy**: Each block processes `BLOCK_SIZE_B` batch elements
2. **E-dimension Loop**: Processes intermediate dimension E in chunks of `BLOCK_SIZE_E`
3. **Fused Computation**: Combines both matmuls and activation in a single pass
4. **Accumulation**: Output is accumulated across E-dimension chunks

**Grid Layout**: `(ceil(B / BLOCK_SIZE_B) * H,)`

#### Backward Kernel (`flashmlp_bwd_kernel`)

The backward kernel uses a clever unified approach:

1. **Split Computation**: Single kernel handles both dX and dW gradients
2. **Program ID Partitioning**:
   - First `ceil(E / BLOCK_SIZE_E)` programs per head compute dW1, dW2
   - Remaining `ceil(B / BLOCK_SIZE_B)` programs compute dX
3. **Recomputation**: Recomputes forward activations to save memory (checkpointing)

**Grid Layout**: `((ceil(B / BLOCK_SIZE_B) + ceil(E / BLOCK_SIZE_E)) * H,)`

**Gradient Formulas**:
- `dX = dZ @ W1^T` where `dZ = dH ⊙ activation'(Z)`
- `dW1 = X^T @ dZ`
- `dW2 = H^T @ dO`
- `dH = dO @ W2^T`

## Usage

### Basic Usage

```python
import torch
from flashmlp import FlashMLP

# Create model
model = FlashMLP(
    num_heads=12,
    hidden_dim=64,
    intermediate_dim=768,
    activation="silu",
    dtype=torch.float16,
    device='cuda'
)

# Forward pass
x = torch.randn(12, 1024, 64, device='cuda', dtype=torch.float16)
output = model(x)  # (12, 1024, 64)

# Backward pass (automatic with autograd)
loss = output.sum()
loss.backward()
```

### Low-Level API

```python
from flashmlp import flashmlp_fwd, flashmlp_bwd, FlashMLPFunction

# Manual forward/backward
x = torch.randn(12, 1024, 64, device='cuda', dtype=torch.float16)
w1 = torch.randn(12, 64, 768, device='cuda', dtype=torch.float16)
w2 = torch.randn(12, 768, 64, device='cuda', dtype=torch.float16)

# Forward
output = flashmlp_fwd(x, w1, w2, activation="silu")

# Backward
do = torch.randn_like(output)
dx, dw1, dw2 = flashmlp_bwd(x, w1, w2, do, activation="silu")

# Or use autograd function
output = FlashMLPFunction.apply(x, w1, w2, "silu")
```

## Supported Activations

- `""` - No activation (linear)
- `"silu"` - SiLU / Swish activation: `x * sigmoid(x)`
- `"leaky_relu"` - Leaky ReLU with slope 0.01
- `"sigmoid"` - Sigmoid activation

## Performance Optimizations

1. **Autotuning**: Automatically selects optimal block sizes for your hardware
2. **Block-based Tiling**: Optimizes memory access patterns for GPU cache
3. **Fused Operations**: Reduces memory bandwidth by fusing matmul + activation
4. **Efficient Recomputation**: Trades computation for memory in backward pass
5. **Multi-stage Pipelines**: Overlaps memory access with computation

### Autotuned Configurations

**Forward Kernel**:
- `BLOCK_SIZE_B` ∈ {32, 64, 128}
- `BLOCK_SIZE_E` ∈ {32, 64}
- `num_stages` ∈ {2, 3, 4}

**Backward Kernel**:
- `BLOCK_SIZE_B` ∈ {32, 64}
- `BLOCK_SIZE_E` ∈ {32, 64}
- `num_stages` ∈ {2, 3}

## Testing

The implementation includes comprehensive unit tests:

```bash
python flashmlp.py
```

Tests include:
1. **Forward Pass**: Comparison against PyTorch reference
2. **Backward Pass**: Gradient verification against PyTorch
3. **Autograd Integration**: End-to-end gradient flow
4. **Module Interface**: High-level API testing

## Comparison to Standard MLP

### Memory Usage

**Standard MLP**:
```python
# Requires storing full intermediate activation
Z = X @ W1          # Store Z: (H, B, E)
H = activation(Z)   # Store H: (H, B, E)
O = H @ W2
```
Memory: `2 * H * B * E` for intermediates

**FlashMLP**:
```python
# Processes E in chunks, recomputes in backward
for e_chunk in range(0, E, BLOCK_SIZE_E):
    z_chunk = X @ W1[:, e_chunk:e_chunk+BLOCK_SIZE_E]
    h_chunk = activation(z_chunk)
    O += h_chunk @ W2[e_chunk:e_chunk+BLOCK_SIZE_E, :]
```
Memory: `2 * H * B * BLOCK_SIZE_E` for intermediates

**Memory Savings**: `~(E / BLOCK_SIZE_E)x` reduction

### Computational Efficiency

1. **Fused Kernels**: Eliminates intermediate writes/reads
2. **Better Cache Utilization**: Tiled access patterns
3. **Reduced Kernel Launches**: Single kernel vs multiple operations

## Implementation Notes

### Activation Derivatives

- **SiLU**: `d_silu(x, h) = sigmoid(x) + h * (1 - sigmoid(x))`
- **Leaky ReLU**: `d_leaky_relu(x) = 1.0 if x >= 0 else 0.01`
- **Sigmoid**: `d_sigmoid(h) = h * (1 - h)` where h = sigmoid(x)

### Numerical Precision

- Uses `float32` for accumulation in matmuls
- Converts to target dtype (float16/bfloat16) after activation
- Tolerances in tests: `atol=3e-2, rtol=3e-2` for bfloat16

## Requirements

- PyTorch >= 2.0
- Triton >= 2.0
- CUDA-capable GPU
- NumPy

```bash
uv pip install torch triton numpy setuptools --index-url https://download.pytorch.org/whl/cu121
```

## Credits

Based on the mhmoe (Multi-Head Mixture of Experts) implementation from:
https://github.com/dtadpole/triton-playground

## License

MIT License