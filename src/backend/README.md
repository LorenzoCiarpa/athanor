# src/backend

Device runtimes. The GPU runtime landed in step 6, for CUDA, written once for NVIDIA and AMD; the HIP build arrives in step 11. Code above the backend includes only `backend.h`, never a vendor header, and passes the device explicitly (the code has no globals).

| File | Content | Step |
| --- | --- | --- |
| `backend.h` | the interface: devices and their properties, device and pinned host memory, copies, memset, three streams per device, events for ordering and timing, the device arena; the matmul library's handle and workspace in `ath_device` (step 7) | 6, 7 |
| `gpu.cu` | the CUDA implementation (HIP through `compat.h`), in the C subset of CUDA C++ with C linkage; built by `make cuda`; creates the cuBLASLt handle and workspace with the device (step 7) | 6, 7 |
| `gpu.h` | private to GPU code (`.cu` files): runtime errors mapped to a status (`ath_gpu_fail`, `ATH_GPU_CHECK`), the device made current, its streams, the check after a kernel launch (`ath_gpu_launched`) | 7 |
| `gpu_none.c` | the same interface in a build without a GPU backend (`make`): no devices, every call `ATH_ERR_UNSUPPORTED` | 6 |
| `compat.h` | the CUDA runtime and cuBLASLt names used by GPU code, mapped to HIP and hipBLASLt; `ATH_WARP`, `ath_shfl_xor` | 6, 7 (HIP half compiled from step 11) |
| `devarena.c` | linear allocation in device memory: blocks from the runtime, mark and reset, block records in a host arena | 6 |
| `cpu.c` | CPU feature detection (`cpuid`) for an instruction-set-specific path | when one lands (`docs/proposals/cpu-matmul-avx512.md`) |

The cuBLASLt handle joined `backend.h` with its first user, the GPU matmul (step 7); graph capture joins it with fast decode (step 9).

## Building

```
make cuda                          # build/cuda; CUDA_HOME (default /usr/local/cuda), CUDA_ARCH (default 120)
make cuda test                     # every suite of the CUDA build, test_backend on the GPU
make cuda DEBUG=1 test             # build/cuda-debug, with ASan and UBSan in host code
./build/cuda/athanor devices --bench
```

- **Toolkit.** `sm_120` (RTX 50 series) needs CUDA 12.8 or newer. `make cuda` checks the version (`tools/dev/check_nvcc.sh`) and names `CUDA_HOME` if it is too old. A toolkit installed outside `/usr/local/cuda`, for example one unpacked from NVIDIA's redist archives without root, is selected with `CUDA_HOME`.
- **Linking.** The CUDA runtime is linked statically (`libcudart_static.a`) and loads the driver itself. A runtime newer than the driver works through CUDA's minor-version compatibility: runtime 13.5 on the CUDA 13.0 driver of the development machine. Device code is compiled to machine code for one architecture, with no PTX to JIT-compile.
- **cuBLASLt** (step 7) is linked as a shared library, `libcublasLt.so.13`, found at run time through the toolkit's `lib64` recorded in the binary (`-rpath`), so `ldd build/cuda/athanor` shows it next to libc and libm (it brings librt, libpthread and libdl, which are part of glibc). Linked statically it would add about 560 MB to every binary: measured with CUDA 13.5, a program that makes one bf16 matmul is 585 MB with `libcublasLt_static.a` and 16 KB with the shared library, because the static archive carries the kernels for every architecture and the linker cannot drop them. `tools/dev/check_nvcc.sh` checks that the toolkit has it. The CPU build is unchanged.
- **Kernel launches without libstdc++.** The launch stub nvcc writes for each kernel keeps the kernel's address in a function-local static, which C++ initializes under a lock from libstdc++ (`__cxa_guard_acquire`). The binaries are linked as C, so kernels are compiled with `-fno-threadsafe-statics`: the value is a constant, and two threads that initialize it at once write the same thing.
- **Object names.** A kernel file has its op's name (`matmul.c`, `matmul.cu`), so `.cu` files compile to `.cu.o`.
- **Strictness.** Device code builds with `-Werror all-warnings` and the host compiler's `-Wall -Wextra -Werror`, as C code does, plus `-fno-exceptions -fno-rtti`.
- **AddressSanitizer.** ASan reserves the address range the NVIDIA driver maps into, so the CUDA debug build runs its tests with `ASAN_OPTIONS=protect_shadow_gap=0`. The Makefile sets it.

## The interface

```c
ath_device d;
ATH_CHECK(ath_device_open(&d, 0));                     // properties and three streams
void *w;  ATH_CHECK(ath_dev_alloc(&d, bytes, &w));
void *p;  ATH_CHECK(ath_pinned_alloc(&d, bytes, &p));  // fast, truly asynchronous host copies
ATH_CHECK(ath_copy(&d, w, p, bytes, ATH_STREAM_H2D));  // direction from the addresses
ath_event up;  ATH_CHECK(ath_event_create(&d, &up, false));
ATH_CHECK(ath_event_record(&d, &up, ATH_STREAM_H2D));
ATH_CHECK(ath_stream_wait(&d, ATH_STREAM_COMPUTE, &up));   // compute starts after the upload, host not blocked
...
ATH_CHECK(ath_stream_sync(&d, ATH_STREAM_COMPUTE));
```

- Every copy and memset is asynchronous with respect to the host. Work on one stream runs in order; work on different streams is ordered only by events, never by synchronizing the whole device.
- **Errors.** Runtime errors become `ATH_ERR_DEVICE` (`ATH_ERR_NOMEM` when memory runs out) with the runtime's message logged, and the runtime's sticky "last error" is cleared, so that the error is not reported again by an unrelated call.
- **Several devices.** Every call selects the device its handle names first, because the runtime keeps a current device per host thread.
- **Device arena.** Small and frequent device allocations go through `ath_dev_arena`, so that the runtime's slow allocator is called once per block (256 MiB by default).
- **The matmul library.** `ath_device_open` also creates a cuBLASLt handle and a 32 MiB workspace (`ATH_BLAS_WORKSPACE`, what cuBLASLt's documentation recommends from Hopper on); `ath_device_close` frees them. One workspace per device is enough because every op runs on the compute stream, one after the other.
- **Kernel launchers** (`src/ops/*.cu`) include `gpu.h`: `ath_gpu_use` makes the device current, the kernels go on `ath_gpu_stream(d, ATH_STREAM_COMPUTE)`, and `ath_gpu_launched` reports an invalid launch at once; a fault while a kernel runs is reported by the next call that waits for the stream.

## Streams

Three per device: compute, host to device, device to host. Every op of `src/ops` runs on the compute stream (step 7). Expert prefetch and hot/cold swaps (steps 12 and phase 4) use the copy streams and synchronize with events.

## One kernel source for two vendors

Kernels are written once in CUDA C under `src/ops/`, as `.cu` files (step 7; `src/ops/README.md`, GPU kernels). `nvcc` compiles them for NVIDIA. `hipcc` compiles the same files for AMD, where `compat.h` maps the `cuda*` names to `hip*` and the `cublasLt*` names to `hipblasLt*`; every runtime or library call GPU code uses must have its line in `compat.h`. Warp exchanges go through `ath_shfl_xor` (`__shfl_xor_sync` on CUDA, `__shfl_xor` on HIP). The cuBLASLt alignment preferences have no hipBLASLt twin and sit behind `#if !defined(__HIP_PLATFORM_AMD__)` in `src/ops/matmul.cu`. Vendor-specific paths (Blackwell FP4/FP8 MMA, RDNA WMMA) sit behind `#if` blocks inside the op that needs them.

| Concern | NVIDIA | AMD |
| --- | --- | --- |
| Wave size | 32 | 32 on RDNA (wave32 mode) |
| Dense GEMM | cuBLASLt | hipBLASLt |
| Tensor-core MMA | `mma.sync`; FP8/FP4 on `sm_120` | WMMA on RDNA3/RDNA4 |
| Graphs | CUDA graphs | HIP graphs |
| Targets | `sm_120` first, then `sm_89`, `sm_86` | `gfx1201`, `gfx1100` |

## Tests

`tests/test_backend.c`. In the CPU build it checks that the stub refuses everything cleanly and skips the GPU tests. In the CUDA build it runs on device 0 and checks every byte:
- copies from pinned and ordinary memory at odd sizes and offsets, and memset;
- the three streams in a pipeline ordered only by events, over 8 rounds that overwrite each other's buffers;
- event timing;
- an allocation larger than the device;
- the device arena.

compute-sanitizer (memcheck with leak check, racecheck) reports no memory errors, leaks or races. Its only reports are the two API errors of the impossible allocation, which the test causes on purpose.
