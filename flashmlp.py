"""
FlashMLP: Memory-efficient Multi-Head MLP with Triton kernels

This module implements a fused MLP operation with support for multiple heads:
    Z = X @ W1
    H = activation(Z)
    O = H @ W2

Shapes:
    X: (H, B, D) - H heads, B batch size, D hidden dimension
    W1: (H, D, E) - First weight matrix
    W2: (H, E, D) - Second weight matrix
    O: (H, B, D) - Output
"""

import torch
import triton
import triton.language as tl
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# Activation Functions
# ============================================================================

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def d_leaky_relu(x):
    return tl.where(x >= 0, 1.0, 0.01)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def d_silu(x, h):
    """Derivative of SiLU given input x and output h = silu(x)"""
    sig = tl.sigmoid(x)
    return sig + h * (1 - sig)


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
    ],
    key=['B', 'D', 'E'],
)
@triton.jit
def flashmlp_fwd_kernel(
    x_ptr, w1_ptr, w2_ptr, o_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_ob, stride_od,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Forward kernel for FlashMLP.

    Computes: O = activation(X @ W1) @ W2

    Each block processes BLOCK_SIZE_B batch elements and loops over E dimension
    in chunks of BLOCK_SIZE_E.
    """
    pid = tl.program_id(axis=0)
    batch_groups = tl.cdiv(B, BLOCK_SIZE_B)
    pid_b = pid % batch_groups
    pid_h = pid // batch_groups

    TARGET_TYPE = x_ptr.type.element_ty

    # Create block pointers for input X
    x_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(B * H, D),
        strides=(stride_xb, stride_xd),
        offsets=(pid_h * B + pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, D),
        order=(1, 0),
    )

    # Create block pointers for weights
    w1_ptrs = tl.make_block_ptr(
        base=w1_ptr,
        shape=(D * H, E),
        strides=(stride_w1d, stride_w1e),
        offsets=(pid_h * D, 0),
        block_shape=(D, BLOCK_SIZE_E),
        order=(1, 0),
    )

    w2_ptrs = tl.make_block_ptr(
        base=w2_ptr,
        shape=(E * H, D),
        strides=(stride_w2e, stride_w2d),
        offsets=(pid_h * E, 0),
        block_shape=(BLOCK_SIZE_E, D),
        order=(1, 0),
    )

    o_ptrs = tl.make_block_ptr(
        base=o_ptr,
        shape=(B * H, D),
        strides=(stride_ob, stride_od),
        offsets=(pid_h * B + pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, D),
        order=(1, 0),
    )

    # Load input
    x = tl.load(x_ptrs)  # (BLOCK_SIZE_B, D)
    o = tl.zeros((BLOCK_SIZE_B, D), dtype=tl.float32)

    # Loop over E dimension
    for e in range(0, tl.cdiv(E, BLOCK_SIZE_E)):
        w1 = tl.load(w1_ptrs)  # (D, BLOCK_SIZE_E)
        w2 = tl.load(w2_ptrs)  # (BLOCK_SIZE_E, D)

        # First matmul: X @ W1
        z = tl.dot(x, w1, out_dtype=tl.float32)  # (BLOCK_SIZE_B, BLOCK_SIZE_E)

        # Apply activation
        if ACTIVATION == "leaky_relu":
            h = leaky_relu(z).to(TARGET_TYPE)
        elif ACTIVATION == "silu":
            h = silu(z).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            h = tl.sigmoid(z).to(TARGET_TYPE)
        else:
            h = z.to(TARGET_TYPE)

        # Second matmul and accumulate: H @ W2
        o = tl.dot(h, w2, o, out_dtype=tl.float32)  # (BLOCK_SIZE_B, D)

        # Advance pointers
        w1_ptrs = tl.advance(w1_ptrs, (0, BLOCK_SIZE_E))
        w2_ptrs = tl.advance(w2_ptrs, (BLOCK_SIZE_E, 0))

    # Store output
    o = o.to(TARGET_TYPE)
    tl.store(o_ptrs, o)


# ============================================================================
# Backward Kernels
# ============================================================================

@triton.jit
def _flashmlp_bwd_dx(
    dx,
    pid_h, pid_b,
    x_ptr, w1_ptr, w2_ptr, do_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_dob, stride_dod,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """
    Backward kernel for computing dX.

    Computes: dX = dZ @ W1^T where dZ = dH * activation'(Z) and dH = dO @ W2^T
    """
    TARGET_TYPE = x_ptr.type.element_ty

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_SIZE_E)

    # Compute pointers
    x_ptrs = x_ptr + ((pid_h * B + pid_b * BLOCK_SIZE_B + offs_b[:, None]) * stride_xb + offs_d[None, :] * stride_xd)
    x_mask = (offs_b[:, None] < B - pid_b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

    do_ptrs = do_ptr + ((pid_h * B + pid_b * BLOCK_SIZE_B + offs_b[:, None]) * stride_dob + offs_d[None, :] * stride_dod)
    do_mask = (offs_b[:, None] < B - pid_b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

    w1_ptrs = w1_ptr + ((pid_h * D + offs_d[:, None]) * stride_w1d + offs_e[None, :] * stride_w1e)
    w2_ptrs = w2_ptr + ((pid_h * E + offs_e[:, None]) * stride_w2e + offs_d[None, :] * stride_w2d)

    # Load input and gradient
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)   # (BLOCK_SIZE_B, D)
    do = tl.load(do_ptrs, mask=do_mask, other=0.0)  # (BLOCK_SIZE_B, D)

    # Loop over E dimension
    for e in range(0, tl.cdiv(E, BLOCK_SIZE_E)):
        w1_mask = (offs_d[:, None] < D) & (offs_e[None, :] < E - e * BLOCK_SIZE_E)
        w2_mask = (offs_e[:, None] < E - e * BLOCK_SIZE_E) & (offs_d[None, :] < D)

        w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)  # (D, BLOCK_SIZE_E)
        w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)  # (BLOCK_SIZE_E, D)

        # Recompute forward pass
        z = tl.dot(x, w1, out_dtype=tl.float32)  # (BLOCK_SIZE_B, BLOCK_SIZE_E)

        if ACTIVATION == "leaky_relu":
            h = leaky_relu(z).to(TARGET_TYPE)
        elif ACTIVATION == "silu":
            h = silu(z).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            h = tl.sigmoid(z).to(TARGET_TYPE)
        else:
            h = z.to(TARGET_TYPE)

        # Backward through second matmul
        dh = tl.dot(do, tl.trans(w2), out_dtype=tl.float32)  # (BLOCK_SIZE_B, BLOCK_SIZE_E)

        # Backward through activation
        if ACTIVATION == "leaky_relu":
            dz = (dh * d_leaky_relu(z)).to(TARGET_TYPE)
        elif ACTIVATION == "silu":
            dz = (dh * d_silu(z, h)).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            sig = h  # sigmoid output is already computed
            dz = (dh * sig * (1 - sig)).to(TARGET_TYPE)
        else:
            dz = dh.to(TARGET_TYPE)

        # Backward through first matmul
        dx += tl.dot(dz, tl.trans(w1), out_dtype=tl.float32)  # (BLOCK_SIZE_B, D)

        # Advance pointers
        w1_ptrs += BLOCK_SIZE_E * stride_w1e
        w2_ptrs += BLOCK_SIZE_E * stride_w2e

    return dx


@triton.jit
def _flashmlp_bwd_dw1w2(
    dw1, dw2,
    pid_h, pid_e,
    x_ptr, w1_ptr, w2_ptr, do_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_dob, stride_dod,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """
    Backward kernel for computing dW1 and dW2.

    Computes: dW1 = X^T @ dZ, dW2 = H^T @ dO
    """
    TARGET_TYPE = x_ptr.type.element_ty

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_SIZE_E)

    # Compute pointers
    x_ptrs = x_ptr + ((pid_h * B + offs_b[:, None]) * stride_xb + offs_d[None, :] * stride_xd)
    do_ptrs = do_ptr + ((pid_h * B + offs_b[:, None]) * stride_dob + offs_d[None, :] * stride_dod)

    w1_ptrs = w1_ptr + ((pid_h * D + offs_d[:, None]) * stride_w1d + (pid_e * BLOCK_SIZE_E + offs_e[None, :]) * stride_w1e)
    w1_mask = (offs_d[:, None] < D) & (offs_e[None, :] < E - pid_e * BLOCK_SIZE_E)

    w2_ptrs = w2_ptr + ((pid_h * E + pid_e * BLOCK_SIZE_E + offs_e[:, None]) * stride_w2e + offs_d[None, :] * stride_w2d)
    w2_mask = (offs_e[:, None] < E - pid_e * BLOCK_SIZE_E) & (offs_d[None, :] < D)

    # Load weights (needed for recomputation)
    w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)  # (D, BLOCK_SIZE_E)
    w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)  # (BLOCK_SIZE_E, D)

    # Loop over batch dimension
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        x_mask = (offs_b[:, None] < B - b * BLOCK_SIZE_B) & (offs_d[None, :] < D)
        do_mask = (offs_b[:, None] < B - b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)   # (BLOCK_SIZE_B, D)
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)  # (BLOCK_SIZE_B, D)

        # Recompute forward pass
        z = tl.dot(x, w1, out_dtype=tl.float32)  # (BLOCK_SIZE_B, BLOCK_SIZE_E)

        if ACTIVATION == "leaky_relu":
            h = leaky_relu(z).to(TARGET_TYPE)
        elif ACTIVATION == "silu":
            h = silu(z).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            h = tl.sigmoid(z).to(TARGET_TYPE)
        else:
            h = z.to(TARGET_TYPE)

        # Backward through second matmul
        dh = tl.dot(do, tl.trans(w2), out_dtype=tl.float32)  # (BLOCK_SIZE_B, BLOCK_SIZE_E)

        # Gradient for W2
        dw2 += tl.dot(tl.trans(h), do, out_dtype=tl.float32)  # (BLOCK_SIZE_E, D)

        # Backward through activation
        if ACTIVATION == "leaky_relu":
            dz = (dh * d_leaky_relu(z)).to(TARGET_TYPE)
        elif ACTIVATION == "silu":
            dz = (dh * d_silu(z, h)).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            sig = h
            dz = (dh * sig * (1 - sig)).to(TARGET_TYPE)
        else:
            dz = dh.to(TARGET_TYPE)

        # Gradient for W1
        dw1 += tl.dot(tl.trans(x), dz, out_dtype=tl.float32)  # (D, BLOCK_SIZE_E)

        # Advance pointers
        x_ptrs += BLOCK_SIZE_B * stride_xb
        do_ptrs += BLOCK_SIZE_B * stride_dob

    return dw1, dw2


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
    ],
    key=['H', 'B', 'D', 'E'],
)
@triton.jit
def flashmlp_bwd_kernel(
    x_ptr, w1_ptr, w2_ptr, dx_ptr, dw1_ptr, dw2_ptr, do_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_dxb, stride_dxd,
    stride_dw1d, stride_dw1e,
    stride_dw2e, stride_dw2d,
    stride_dob, stride_dod,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """
    Unified backward kernel that computes both dX and dW1/dW2.

    Uses a clever grid layout where some blocks compute dX and others compute dW.
    """
    pid = tl.program_id(axis=0)

    batch_groups_e = tl.cdiv(E, BLOCK_SIZE_E)
    batch_groups_b = tl.cdiv(B, BLOCK_SIZE_B)
    idx = pid % (batch_groups_e + batch_groups_b)
    pid_h = pid // (batch_groups_e + batch_groups_b)

    TARGET_TYPE = x_ptr.type.element_ty

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_SIZE_E)

    if idx >= batch_groups_e:
        # Compute dX
        pid_b = idx - batch_groups_e

        dx_ptrs = dx_ptr + ((pid_h * B + pid_b * BLOCK_SIZE_B + offs_b[:, None]) * stride_dxb + offs_d[None, :] * stride_dxd)
        dx_mask = (offs_b[:, None] < B - pid_b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

        dx = tl.zeros((BLOCK_SIZE_B, D), dtype=tl.float32)
        dx = _flashmlp_bwd_dx(
            dx, pid_h, pid_b,
            x_ptr, w1_ptr, w2_ptr, do_ptr,
            H, B, D, E,
            stride_xb, stride_xd,
            stride_w1d, stride_w1e,
            stride_w2e, stride_w2d,
            stride_dob, stride_dod,
            BLOCK_SIZE_B, BLOCK_SIZE_E,
            ACTIVATION
        )

        tl.store(dx_ptrs, dx.to(TARGET_TYPE), mask=dx_mask)

    else:
        # Compute dW1 and dW2
        pid_e = idx

        dw1_ptrs = dw1_ptr + ((pid_h * D + offs_d[:, None]) * stride_dw1d + (pid_e * BLOCK_SIZE_E + offs_e[None, :]) * stride_dw1e)
        dw1_mask = (offs_d[:, None] < D) & (offs_e[None, :] < E - pid_e * BLOCK_SIZE_E)

        dw2_ptrs = dw2_ptr + ((pid_h * E + pid_e * BLOCK_SIZE_E + offs_e[:, None]) * stride_dw2e + offs_d[None, :] * stride_dw2d)
        dw2_mask = (offs_e[:, None] < E - pid_e * BLOCK_SIZE_E) & (offs_d[None, :] < D)

        dw1 = tl.zeros((D, BLOCK_SIZE_E), dtype=tl.float32)
        dw2 = tl.zeros((BLOCK_SIZE_E, D), dtype=tl.float32)

        dw1, dw2 = _flashmlp_bwd_dw1w2(
            dw1, dw2, pid_h, pid_e,
            x_ptr, w1_ptr, w2_ptr, do_ptr,
            H, B, D, E,
            stride_xb, stride_xd,
            stride_w1d, stride_w1e,
            stride_w2e, stride_w2d,
            stride_dob, stride_dod,
            BLOCK_SIZE_B, BLOCK_SIZE_E,
            ACTIVATION
        )

        tl.store(dw1_ptrs, dw1.to(TARGET_TYPE), mask=dw1_mask)
        tl.store(dw2_ptrs, dw2.to(TARGET_TYPE), mask=dw2_mask)


# ============================================================================
# Python Wrappers
# ============================================================================

def flashmlp_fwd(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                 activation: str = "") -> torch.Tensor:
    """
    FlashMLP forward pass.

    Args:
        x: Input tensor of shape (H, B, D)
        w1: First weight matrix of shape (H, D, E)
        w2: Second weight matrix of shape (H, E, D)
        activation: Activation function ("silu", "leaky_relu", "sigmoid", or "")

    Returns:
        Output tensor of shape (H, B, D)
    """
    # Check constraints
    assert x.shape[0] == w1.shape[0], "Number of heads must match"
    assert x.shape[0] == w2.shape[0], "Number of heads must match"
    assert x.shape[2] == w1.shape[1], "Hidden dimension mismatch"
    assert w1.shape[2] == w2.shape[1], "Intermediate dimension mismatch"
    assert x.shape[2] == w2.shape[2], "Output dimension mismatch"

    H, B, D = x.shape
    E = w1.shape[2]

    # Reshape for kernel
    x = x.reshape(H * B, D)
    w1 = w1.reshape(D * H, E)
    w2 = w2.reshape(E * H, D)

    assert x.is_contiguous(), "Input X must be contiguous"
    assert w1.is_contiguous(), "Weight W1 must be contiguous"
    assert w2.is_contiguous(), "Weight W2 must be contiguous"

    # Allocate output
    o = torch.zeros_like(x)

    # Launch kernel
    grid = lambda META: (triton.cdiv(B, META['BLOCK_SIZE_B']) * H,)

    flashmlp_fwd_kernel[grid](
        x, w1, w2, o,
        H, B, D, E,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        o.stride(0), o.stride(1),
        ACTIVATION=activation
    )

    return o.reshape(H, B, D)


def flashmlp_bwd(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                 do: torch.Tensor, activation: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashMLP backward pass.

    Args:
        x: Input tensor of shape (H, B, D)
        w1: First weight matrix of shape (H, D, E)
        w2: Second weight matrix of shape (H, E, D)
        do: Output gradient of shape (H, B, D)
        activation: Activation function ("silu", "leaky_relu", "sigmoid", or "")

    Returns:
        Tuple of (dx, dw1, dw2) gradients
    """
    # Check constraints
    assert x.shape[2] == w1.shape[1], "Hidden dimension mismatch"
    assert w1.shape[2] == w2.shape[1], "Intermediate dimension mismatch"
    assert x.shape[2] == w2.shape[2], "Output dimension mismatch"
    assert x.shape == do.shape, "Gradient shape mismatch"

    H, B, D = x.shape
    E = w1.shape[2]

    # Reshape for kernel
    x = x.reshape(H * B, D)
    w1 = w1.reshape(D * H, E)
    w2 = w2.reshape(E * H, D)
    do = do.reshape(H * B, D).contiguous()

    assert x.is_contiguous(), "Input X must be contiguous"
    assert w1.is_contiguous(), "Weight W1 must be contiguous"
    assert w2.is_contiguous(), "Weight W2 must be contiguous"
    assert do.is_contiguous(), "Gradient dO must be contiguous"

    # Allocate gradients
    dx = torch.zeros_like(x)
    dw1 = torch.zeros_like(w1)
    dw2 = torch.zeros_like(w2)

    # Launch kernel
    grid = lambda META: (
        (triton.cdiv(B, META['BLOCK_SIZE_B']) + triton.cdiv(E, META['BLOCK_SIZE_E'])) * H,
    )

    flashmlp_bwd_kernel[grid](
        x, w1, w2, dx, dw1, dw2, do,
        H, B, D, E,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        dx.stride(0), dx.stride(1),
        dw1.stride(0), dw1.stride(1),
        dw2.stride(0), dw2.stride(1),
        do.stride(0), do.stride(1),
        ACTIVATION=activation
    )

    return dx.reshape(H, B, D), dw1.reshape(H, D, E), dw2.reshape(H, E, D)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class FlashMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, activation=""):
        o = flashmlp_fwd(x, w1, w2, activation)
        ctx.save_for_backward(x, w1, w2)
        ctx.activation = activation
        return o

    @staticmethod
    def backward(ctx, do):
        x, w1, w2 = ctx.saved_tensors
        dx, dw1, dw2 = flashmlp_bwd(x, w1, w2, do, ctx.activation)
        return dx, dw1, dw2, None


# ============================================================================
# High-level API
# ============================================================================

class FlashMLP(torch.nn.Module):
    """
    Multi-Head MLP module using FlashMLP kernels.

    Args:
        num_heads: Number of heads
        hidden_dim: Hidden dimension (D)
        intermediate_dim: Intermediate dimension (E)
        activation: Activation function ("silu", "leaky_relu", "sigmoid", or "")
        dtype: Data type for weights
        device: Device to create weights on
    """

    def __init__(self, num_heads: int, hidden_dim: int, intermediate_dim: int,
                 activation: str = "silu", dtype=torch.float16, device='cuda'):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        # Initialize weights
        self.w1 = torch.nn.Parameter(
            torch.randn(num_heads, hidden_dim, intermediate_dim, dtype=dtype, device=device) / np.sqrt(intermediate_dim)
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(num_heads, intermediate_dim, hidden_dim, dtype=dtype, device=device) / np.sqrt(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (H, B, D) or (B, D) which will be expanded to (H, B, D)

        Returns:
            Output of shape (H, B, D) or (B, D) depending on input
        """
        if x.dim() == 2:
            # Expand to (H, B, D)
            x = x.unsqueeze(0).expand(self.num_heads, -1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        o = FlashMLPFunction.apply(x, self.w1, self.w2, self.activation)

        if squeeze_output:
            # Average across heads and return (B, D)
            o = o.mean(dim=0)

        return o


class PyTorchMLP(torch.nn.Module):
    """
    Multi-Head MLP module using standard PyTorch operations.

    Same interface as FlashMLP for direct comparison.
    """

    def __init__(self, num_heads: int, hidden_dim: int, intermediate_dim: int,
                 activation: str = "silu", dtype=torch.float16, device='cuda'):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        self.w1 = torch.nn.Parameter(
            torch.randn(num_heads, hidden_dim, intermediate_dim, dtype=dtype, device=device) / np.sqrt(intermediate_dim)
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(num_heads, intermediate_dim, hidden_dim, dtype=dtype, device=device) / np.sqrt(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.num_heads, -1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        z = torch.bmm(x, self.w1)

        if self.activation == "leaky_relu":
            h = torch.nn.functional.leaky_relu(z, negative_slope=0.01)
        elif self.activation == "silu":
            h = torch.nn.functional.silu(z)
        elif self.activation == "sigmoid":
            h = torch.sigmoid(z)
        else:
            h = z

        o = torch.bmm(h, self.w2)

        if squeeze_output:
            o = o.mean(dim=0)

        return o


# ============================================================================
# Reference PyTorch Implementation (for testing)
# ============================================================================

def flashmlp_torch_fwd(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                       activation: str = "") -> torch.Tensor:
    """Reference PyTorch implementation for testing."""
    z = torch.bmm(x, w1)

    if activation == "leaky_relu":
        h = torch.nn.functional.leaky_relu(z, negative_slope=0.01)
    elif activation == "silu":
        h = torch.nn.functional.silu(z)
    elif activation == "sigmoid":
        h = torch.sigmoid(z)
    else:
        h = z

    o = torch.bmm(h, w2)
    return o


def flashmlp_torch_bwd(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                       do: torch.Tensor, activation: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference PyTorch backward implementation for testing."""
    # Forward pass
    z = torch.bmm(x, w1)

    if activation == "leaky_relu":
        h = torch.nn.functional.leaky_relu(z, negative_slope=0.01)
    elif activation == "silu":
        h = torch.nn.functional.silu(z)
    elif activation == "sigmoid":
        h = torch.sigmoid(z)
    else:
        h = z

    # Backward pass
    dh = torch.bmm(do, torch.transpose(w2, -1, -2))  # (H, B, E)
    dw2 = torch.bmm(torch.transpose(h, -1, -2), do)  # (H, E, D)

    # Backward through activation
    if activation == "leaky_relu":
        dz = dh * torch.where(z >= 0, 1.0, 0.01).to(dh.dtype)
    elif activation == "silu":
        sig = torch.sigmoid(z)
        dz = dh * (sig + z * sig * (1 - sig))
    elif activation == "sigmoid":
        sig = h
        dz = dh * sig * (1 - sig)
    else:
        dz = dh

    dx = torch.bmm(dz, torch.transpose(w1, -1, -2))  # (H, B, D)
    dw1 = torch.bmm(torch.transpose(x, -1, -2), dz)  # (H, D, E)

    return dx, dw1, dw2


# ============================================================================
# Unit Tests
# ============================================================================

def test_forward():
    """Test FlashMLP forward pass against PyTorch reference."""
    print("=" * 80)
    print("Testing FlashMLP Forward Pass")
    print("=" * 80)

    torch.manual_seed(42)
    dtype = torch.bfloat16
    H, B, D, E = 12, 1024, 64, 768

    x = torch.randn((H, B, D), device='cuda', dtype=dtype) / np.sqrt(D)
    w1 = torch.randn((H, D, E), device='cuda', dtype=dtype) / np.sqrt(E)
    w2 = torch.randn((H, E, D), device='cuda', dtype=dtype) / np.sqrt(D)

    for activation in ["", "silu", "leaky_relu", "sigmoid"]:
        print(f"\nActivation: {activation or 'none'}")

        triton_output = flashmlp_fwd(x, w1, w2, activation=activation)
        torch_output = flashmlp_torch_fwd(x, w1, w2, activation=activation)

        diff = torch.abs(triton_output - torch_output).float()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")


def test_backward():
    """Test FlashMLP backward pass against PyTorch reference."""
    print("\n" + "=" * 80)
    print("Testing FlashMLP Backward Pass")
    print("=" * 80)

    torch.manual_seed(42)
    dtype = torch.bfloat16
    H, B, D, E = 12, 1024, 64, 768

    x = torch.randn((H, B, D), device='cuda', dtype=dtype) / np.sqrt(D)
    w1 = torch.randn((H, D, E), device='cuda', dtype=dtype) / np.sqrt(E)
    w2 = torch.randn((H, E, D), device='cuda', dtype=dtype) / np.sqrt(D)
    do = torch.randn((H, B, D), device='cuda', dtype=dtype) / np.sqrt(D)

    for activation in ["", "silu", "leaky_relu"]:
        print(f"\nActivation: {activation or 'none'}")

        triton_dx, triton_dw1, triton_dw2 = flashmlp_bwd(x, w1, w2, do, activation=activation)
        torch_dx, torch_dw1, torch_dw2 = flashmlp_torch_bwd(x, w1, w2, do, activation=activation)

        for name, triton_grad, torch_grad in [("dx", triton_dx, torch_dx),
                                                ("dw1", triton_dw1, torch_dw1),
                                                ("dw2", triton_dw2, torch_dw2)]:
            diff = torch.abs(triton_grad - torch_grad).float()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}", end="")

            if torch.allclose(triton_grad, torch_grad, atol=3e-2, rtol=3e-2):
                print(" ✅")
            else:
                print(" ❌")


def test_autograd():
    """Test FlashMLP autograd integration."""
    print("\n" + "=" * 80)
    print("Testing FlashMLP Autograd Integration")
    print("=" * 80)

    torch.manual_seed(42)
    dtype = torch.bfloat16
    H, B, D, E = 8, 512, 64, 512

    x = (torch.randn((H, B, D), device='cuda', dtype=dtype) / np.sqrt(D)).requires_grad_(True)
    w1 = (torch.randn((H, D, E), device='cuda', dtype=dtype) / np.sqrt(E)).requires_grad_(True)
    w2 = (torch.randn((H, E, D), device='cuda', dtype=dtype) / np.sqrt(D)).requires_grad_(True)

    activation = "silu"

    # FlashMLP
    o_flash = FlashMLPFunction.apply(x, w1, w2, activation)
    loss_flash = o_flash.sum()
    loss_flash.backward()

    dx_flash = x.grad.clone()
    dw1_flash = w1.grad.clone()
    dw2_flash = w2.grad.clone()

    # PyTorch reference
    x.grad = None
    w1.grad = None
    w2.grad = None

    o_torch = flashmlp_torch_fwd(x, w1, w2, activation)
    loss_torch = o_torch.sum()
    loss_torch.backward()

    dx_torch = x.grad
    dw1_torch = w1.grad
    dw2_torch = w2.grad

    print(f"\nForward output match: ", end="")
    if torch.allclose(o_flash, o_torch, atol=1e-2, rtol=1e-2):
        print("✅")
    else:
        print("❌")

    for name, grad_flash, grad_torch in [("dx", dx_flash, dx_torch),
                                          ("dw1", dw1_flash, dw1_torch),
                                          ("dw2", dw2_flash, dw2_torch)]:
        diff = torch.abs(grad_flash - grad_torch).float()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}", end="")

        if torch.allclose(grad_flash, grad_torch, atol=3e-2, rtol=3e-2):
            print(" ✅")
        else:
            print(" ❌")


def test_module():
    """Test FlashMLP module."""
    print("\n" + "=" * 80)
    print("Testing FlashMLP Module")
    print("=" * 80)

    torch.manual_seed(42)
    dtype = torch.bfloat16
    H, B, D, E = 8, 512, 64, 512

    model = FlashMLP(num_heads=H, hidden_dim=D, intermediate_dim=E,
                     activation="silu", dtype=dtype, device='cuda')

    x = torch.randn((H, B, D), device='cuda', dtype=dtype, requires_grad=True)

    # Forward
    o = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {o.shape}")
    print(f"Output mean: {o.mean().item():.6f}")
    print(f"Output std: {o.std().item():.6f}")

    # Backward
    loss = o.sum()
    loss.backward()

    print(f"\nGradients computed successfully:")
    print(f"  x.grad: {x.grad is not None} ✅")
    print(f"  w1.grad: {model.w1.grad is not None} ✅")
    print(f"  w2.grad: {model.w2.grad is not None} ✅")


if __name__ == '__main__':
    test_forward()
    test_backward()
    test_autograd()
    test_module()
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
