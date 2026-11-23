# src/backend

Device runtimes: CPU, and GPU for NVIDIA and AMD from one source. Introduced in step 3 (CPU) and step 6 (GPU).

| File | Content |
| --- | --- |
| `backend.h` | device handle, memory, copies, streams, events, graph capture, capability query |
| `cpu.c` | host memory, CPU feature detection (`cpuid`), dispatch to the core thread pool |
| `gpu.cu` | CUDA/HIP runtime glue, cuBLASLt/hipBLASLt handles |
| `compat.h` | mapping of the CUDA runtime API and intrinsics to HIP, wave-size constant `ATH_WARP` |

## One kernel source for two vendors

Kernels are written once in CUDA C under `src/ops/*.cu`. `nvcc` compiles them for NVIDIA; `hipcc` compiles the same files for AMD with `compat.h` mapping `cudaX` to `hipX`. Vendor-specific paths (Blackwell FP4/FP8 MMA, RDNA WMMA) sit behind `#if` blocks inside the op that needs them.

| Concern | NVIDIA | AMD |
| --- | --- | --- |
| Wave size | 32 | 32 on RDNA (wave32 mode) |
| Dense GEMM | cuBLASLt | hipBLASLt |
| Tensor-core MMA | `mma.sync`; FP8/FP4 on `sm_120` | WMMA on RDNA3/RDNA4 |
| Graphs | CUDA graphs | HIP graphs |
| Targets | `sm_120` first, then `sm_89`, `sm_86` | `gfx1201`, `gfx1100` |

## Streams

Three per device: compute, host-to-device, device-to-host. Expert prefetch and hot/cold swaps use the copy streams and synchronize with events, never with device-wide synchronization.
