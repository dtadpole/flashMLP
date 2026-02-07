"""
Comparison between PyTorch MLP and Triton FlashMLP
"""

import torch
import numpy as np
import time
from flashmlp import FlashMLP, PyTorchMLP


def compare_correctness():
    """Compare outputs and gradients between PyTorch and Triton versions."""
    print("=" * 80)
    print("Correctness Comparison: PyTorch vs Triton")
    print("=" * 80)

    torch.manual_seed(42)
    dtype = torch.bfloat16
    H, B, D, E = 12, 1024, 64, 768

    for activation in ["", "silu", "leaky_relu", "sigmoid"]:
        print(f"\nActivation: {activation or 'none'}")

        # Create models with shared weights
        triton_model = FlashMLP(H, D, E, activation=activation, dtype=dtype, device='cuda')
        pytorch_model = PyTorchMLP(H, D, E, activation=activation, dtype=dtype, device='cuda')
        pytorch_model.w1.data.copy_(triton_model.w1.data)
        pytorch_model.w2.data.copy_(triton_model.w2.data)

        # Create input
        x = (torch.randn(H, B, D, device='cuda', dtype=dtype) / np.sqrt(D)).requires_grad_(True)

        # Forward
        o_triton = triton_model(x)
        loss_triton = o_triton.sum()
        loss_triton.backward()
        dx_triton = x.grad.clone()
        dw1_triton = triton_model.w1.grad.clone()
        dw2_triton = triton_model.w2.grad.clone()

        # Reset grads
        x.grad = None

        o_pytorch = pytorch_model(x)
        loss_pytorch = o_pytorch.sum()
        loss_pytorch.backward()
        dx_pytorch = x.grad
        dw1_pytorch = pytorch_model.w1.grad
        dw2_pytorch = pytorch_model.w2.grad

        # Compare forward
        fwd_diff = torch.abs(o_triton - o_pytorch).float()
        fwd_ok = torch.allclose(o_triton, o_pytorch, atol=1e-2, rtol=1e-2)
        print(f"  Forward:  max_diff={fwd_diff.max().item():.6f}  mean_diff={fwd_diff.mean().item():.6f}  {'PASS' if fwd_ok else 'FAIL'}")

        # Compare gradients
        for name, t_grad, p_grad in [("dx", dx_triton, dx_pytorch),
                                      ("dw1", dw1_triton, dw1_pytorch),
                                      ("dw2", dw2_triton, dw2_pytorch)]:
            diff = torch.abs(t_grad - p_grad).float()
            ok = torch.allclose(t_grad, p_grad, atol=3e-2, rtol=3e-2)
            print(f"  {name:>7s}:  max_diff={diff.max().item():.6f}  mean_diff={diff.mean().item():.6f}  {'PASS' if ok else 'FAIL'}")


def benchmark_timing(label, fn, warmup=10, iters=100):
    """Benchmark a function with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return elapsed


def compare_performance():
    """Benchmark speed and memory for PyTorch vs Triton."""
    print("\n" + "=" * 80)
    print("Performance Comparison: PyTorch vs Triton")
    print("=" * 80)

    dtype = torch.bfloat16
    activation = "silu"

    configs = [
        (8,   512,  64,  512),
        (12,  1024, 64,  768),
        (16,  2048, 64,  1024),
        (32,  4096, 64,  512),
    ]

    print(f"\n{'Config (H,B,D,E)':<24s} {'PyTorch fwd':>14s} {'Triton fwd':>14s} {'Speedup':>10s} {'PyTorch fwd+bwd':>16s} {'Triton fwd+bwd':>16s} {'Speedup':>10s}")
    print("-" * 110)

    for H, B, D, E in configs:
        torch.manual_seed(0)

        triton_model = FlashMLP(H, D, E, activation=activation, dtype=dtype, device='cuda')
        pytorch_model = PyTorchMLP(H, D, E, activation=activation, dtype=dtype, device='cuda')
        pytorch_model.w1.data.copy_(triton_model.w1.data)
        pytorch_model.w2.data.copy_(triton_model.w2.data)

        x = (torch.randn(H, B, D, device='cuda', dtype=dtype) / np.sqrt(D))

        # Forward-only benchmark
        def fwd_pytorch():
            return pytorch_model(x)

        def fwd_triton():
            return triton_model(x)

        t_pytorch_fwd = benchmark_timing("pytorch_fwd", fwd_pytorch)
        t_triton_fwd = benchmark_timing("triton_fwd", fwd_triton)
        speedup_fwd = t_pytorch_fwd / t_triton_fwd

        # Forward + backward benchmark
        def fwd_bwd_pytorch():
            x_in = x.detach().requires_grad_(True)
            o = pytorch_model(x_in)
            o.sum().backward()

        def fwd_bwd_triton():
            x_in = x.detach().requires_grad_(True)
            o = triton_model(x_in)
            o.sum().backward()

        t_pytorch_fwdbwd = benchmark_timing("pytorch_fwd+bwd", fwd_bwd_pytorch)
        t_triton_fwdbwd = benchmark_timing("triton_fwd+bwd", fwd_bwd_triton)
        speedup_fwdbwd = t_pytorch_fwdbwd / t_triton_fwdbwd

        config_str = f"({H},{B},{D},{E})"
        print(f"{config_str:<24s} {t_pytorch_fwd:>11.3f} ms {t_triton_fwd:>11.3f} ms {speedup_fwd:>9.2f}x {t_pytorch_fwdbwd:>13.3f} ms {t_triton_fwdbwd:>13.3f} ms {speedup_fwdbwd:>9.2f}x")

        # Clear caches
        pytorch_model.zero_grad()
        triton_model.zero_grad()


def compare_memory():
    """Compare peak GPU memory usage between PyTorch and Triton."""
    print("\n" + "=" * 80)
    print("Memory Comparison: PyTorch vs Triton (activation=silu, dtype=bfloat16)")
    print("=" * 80)

    dtype = torch.bfloat16
    activation = "silu"

    configs = [
        (8,   512,  64,  512),
        (12,  1024, 64,  768),
        (16,  2048, 64,  1024),
        (32,  4096, 64,  512),
        (12,  4096, 64,  768),
    ]

    print(f"\n{'Config (H,B,D,E)':<24s} {'PyTorch peak':>14s} {'Triton peak':>14s} {'Savings':>10s} {'PyTorch fwd':>14s} {'Triton fwd':>14s} {'Savings':>10s}")
    print("-" * 106)

    for H, B, D, E in configs:
        row = {}
        for name, ModelClass in [("PyTorch", PyTorchMLP), ("Triton", FlashMLP)]:
            torch.cuda.empty_cache()

            model = ModelClass(H, D, E, activation=activation, dtype=dtype, device='cuda')
            x_warmup = (torch.randn(H, B, D, device='cuda', dtype=dtype) / np.sqrt(D)).requires_grad_(True)

            # Warmup: compile Triton kernels / CUDA caches
            o_warmup = model(x_warmup)
            o_warmup.sum().backward()
            del x_warmup, o_warmup
            model.zero_grad()
            torch.cuda.empty_cache()

            # Now measure
            torch.cuda.reset_peak_memory_stats()
            x = (torch.randn(H, B, D, device='cuda', dtype=dtype) / np.sqrt(D)).requires_grad_(True)

            mem_before = torch.cuda.memory_allocated()

            o = model(x)
            fwd_mem = (torch.cuda.memory_allocated() - mem_before) / (1024 ** 2)

            o.sum().backward()
            peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / (1024 ** 2)

            row[name] = (peak_mem, fwd_mem)

            del model, x, o
            torch.cuda.empty_cache()

        pt_peak, pt_fwd = row["PyTorch"]
        tr_peak, tr_fwd = row["Triton"]
        peak_saving = (1 - tr_peak / pt_peak) * 100 if pt_peak > 0 else 0
        fwd_saving = (1 - tr_fwd / pt_fwd) * 100 if pt_fwd > 0 else 0

        config_str = f"({H},{B},{D},{E})"
        print(f"{config_str:<24s} {pt_peak:>11.1f} MB {tr_peak:>11.1f} MB {peak_saving:>8.1f}% {pt_fwd:>11.1f} MB {tr_fwd:>11.1f} MB {fwd_saving:>8.1f}%")


if __name__ == '__main__':
    print("FlashMLP: PyTorch vs Triton Comparison")
    print("Note: Requires a CUDA-capable GPU\n")

    compare_correctness()
    compare_performance()
    compare_memory()

    print("\n" + "=" * 80)
    print("All comparisons completed!")
    print("=" * 80)
