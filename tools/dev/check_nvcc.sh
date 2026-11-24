#!/bin/sh
# check_nvcc.sh <nvcc> <arch>: fail with a clear message unless <nvcc> exists
# and is recent enough for sm_<arch>. Called by `make cuda` (Makefile, cuda-check).
#
# sm_120 (RTX 50 series, Blackwell) needs CUDA 12.8 or newer; older
# architectures are accepted by any CUDA 12.x.
nvcc=$1
arch=$2
if [ ! -x "$nvcc" ]; then
    echo "make cuda: no nvcc at $nvcc; set CUDA_HOME to a CUDA 12.8+ toolkit" >&2
    exit 1
fi
version=$("$nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p')
set -- $version
major=${1:-0}
minor=${2:-0}
need_minor=0
[ "$arch" -ge 120 ] && need_minor=8
if [ "$major" -lt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -lt "$need_minor" ]; }; then
    echo "make cuda: $nvcc is CUDA $major.$minor; sm_$arch needs CUDA 12.$need_minor or newer (set CUDA_HOME)" >&2
    exit 1
fi
