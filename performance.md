# FlashMLP Performance: PyTorch vs Triton

All benchmarks run with `activation=silu`, `dtype=bfloat16`.

## Speed

| Config (H,B,D,E) | PyTorch fwd | Triton fwd | Fwd speedup | PyTorch fwd+bwd | Triton fwd+bwd | Fwd+bwd speedup |
|---|--:|--:|--:|--:|--:|--:|
| (8,512,64,512) | 0.019 ms | 0.036 ms | 0.52x | 0.158 ms | 0.217 ms | 0.73x |
| (12,1024,64,768) | 0.039 ms | 0.035 ms | 1.11x | 0.156 ms | 0.222 ms | 0.70x |
| (16,2048,64,1024) | 0.180 ms | 0.036 ms | **4.96x** | 0.673 ms | 0.219 ms | **3.07x** |
| (32,4096,64,512) | 0.394 ms | 0.060 ms | **6.55x** | 1.140 ms | 0.380 ms | **3.00x** |

### Observations

- At small sizes (H=8, B=512), PyTorch is faster. Kernel launch overhead dominates when there isn't enough work to saturate the GPU.
- At large sizes (H=16+, B=2048+), Triton wins with 5-6.5x forward speedup and 3x fwd+bwd speedup. The fused kernel avoids materializing intermediates and reduces memory bandwidth.
- The crossover point is around (12, 1024, 64, 768) where both are roughly equal on forward.

## Memory

Peak GPU memory measured after warming up Triton JIT to exclude one-time compilation overhead.

| Config (H,B,D,E) | PyTorch peak | Triton peak | Peak savings | PyTorch fwd alloc | Triton fwd alloc | Fwd savings |
|---|--:|--:|--:|--:|--:|--:|
| (8,512,64,512) | 13.5 MB | 2.5 MB | **81.5%** | 8.5 MB | 0.5 MB | **94.1%** |
| (12,1024,64,768) | 58.1 MB | 7.5 MB | **87.1%** | 37.5 MB | 1.5 MB | **96.0%** |
| (16,2048,64,1024) | 202.0 MB | 16.0 MB | **92.1%** | 132.0 MB | 4.0 MB | **97.0%** |
| (32,4096,64,512) | 418.0 MB | 52.0 MB | **87.6%** | 272.0 MB | 16.0 MB | **94.1%** |
| (12,4096,64,768) | 229.1 MB | 21.0 MB | **90.8%** | 150.0 MB | 6.0 MB | **96.0%** |

### Observations

- Triton saves 94-97% of forward memory consistently across all configs. It processes the intermediate dimension E in small blocks instead of materializing the full (H, B, E) tensor.
- Peak memory (including backward pass) savings are 81-92%. The backward kernel recomputes activations rather than storing them, trading compute for memory.
- Memory savings scale with the intermediate dimension E: larger E means PyTorch stores a proportionally larger intermediate tensor, while Triton's usage stays nearly constant per block size.
