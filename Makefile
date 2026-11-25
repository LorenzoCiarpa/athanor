# athanor build. Plain make, no configure step.
#
#   make            CPU build (default), output in build/
#   make test       build and run the test suites
#   make check-alloc  verify that inspect allocates only through arenas
#   make golden     regenerate tests/golden/data from the reference implementation
#   make DEBUG=1    -O0 -g, ATH_ASSERT enabled, address and UB sanitizers
#   make cuda       the same with the CUDA backend, in build/cuda; `make cuda test`
#                   runs every suite of that build (CUDA_HOME and CUDA_ARCH below)
#   make hip        the HIP build (arrives in step 11)
#   make clean

FAMILIES ?= qwen35
# Checkpoint and golden data used by the tests that need them (skipped if absent).
ATH_MODEL_DIR  ?= models/Qwen3.5-0.8B
ATH_GOLDEN_DIR ?= tests/golden/data/qwen35-0.8b
export ATH_MODEL_DIR ATH_GOLDEN_DIR
# Python of the golden-data environment (tests/golden/README.md).
GOLDEN_PYTHON ?= tests/golden/.venv/bin/python
CC       ?= cc

# GPU backend: none (the default) or cuda, chosen by naming the goal `cuda`
# anywhere on the command line, so that `make cuda test` tests the CUDA build.
GPU ?= none
ifneq ($(filter cuda,$(MAKECMDGOALS)),)
GPU := cuda
endif
# CUDA toolkit 12.8 or newer, the first that targets sm_120 (RTX 50 series).
# Set CUDA_HOME if the toolkit is not in /usr/local/cuda.
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 120
NVCC      := $(CUDA_HOME)/bin/nvcc

ifeq ($(GPU),cuda)
BUILD    ?= build/cuda$(if $(filter 1,$(DEBUG)),-debug)
else ifeq ($(DEBUG),1)
BUILD    ?= build/debug
else
BUILD    ?= build
endif

CFLAGS := -std=c11 -Wall -Wextra -Werror -D_GNU_SOURCE -MMD -MP
ifeq ($(DEBUG),1)
CFLAGS  += -O0 -g -DATH_DEBUG -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address,undefined
else
CFLAGS  += -O2
endif
CFLAGS += -Isrc
LDLIBS := -lm -lpthread

# Device code: CUDA C (the C subset of CUDA C++), machine code for one
# architecture. The CUDA runtime is linked statically and loads the driver
# itself. cuBLASLt (the dense matmul) is linked as a shared library found
# through the toolkit's path recorded in the binary: statically it would add
# about 560 MB to every binary (src/backend/README.md, Linking).
# -fno-threadsafe-statics: the launch stub nvcc writes for each kernel keeps
# the kernel's address in a function-local static, which C++ guards with
# a lock from libstdc++; the binary is linked as C, without it. The value is
# a constant, so two threads that set it at once set the same thing.
NVCCFLAGS := -std=c++17 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -Werror all-warnings \
             -Xcompiler -Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-fno-threadsafe-statics -Isrc -MMD -MP
ifeq ($(DEBUG),1)
NVCCFLAGS += -O0 -g -lineinfo -DATH_DEBUG -Xcompiler -fsanitize=address,-fsanitize=undefined,-fno-omit-frame-pointer
else
NVCCFLAGS += -O2 -lineinfo
endif
ifeq ($(GPU),cuda)
LDLIBS += -L$(CUDA_HOME)/lib64 -lcublasLt -Wl,-rpath,$(CUDA_HOME)/lib64 \
          $(CUDA_HOME)/lib64/libcudart_static.a -ldl -lrt
endif

# Families are selected at compile time (docs/ARCHITECTURE.md, rule 1): each
# one is a folder src/models/<name>/ that defines `ath_family_<name>`, and the
# registry learns the list from the generated header $(BUILD)/gen/families.h.
$(foreach f,$(FAMILIES),$(if $(wildcard src/models/$(f)/),,$(error unknown family '$(f)': no src/models/$(f)/)))
FAMILY_SRC := $(foreach f,$(FAMILIES),$(wildcard src/models/$(f)/*.c))
FAMILIES_H := $(BUILD)/gen/families.h
CFLAGS     += -I$(BUILD)/gen

CORE_SRC  := $(wildcard src/core/*.c)
OPS_SRC   := $(wildcard src/ops/*.c)
TEXT_SRC  := $(wildcard src/text/*.c)
MODEL_SRC := $(wildcard src/model/*.c)
# GPU code: the backend's gpu.cu and the kernels of src/ops/*.cu with a GPU;
# without one, the stubs gpu_none.c of both (src/backend/README.md).
BACKEND_SRC := $(filter-out src/backend/gpu_none.c,$(wildcard src/backend/*.c))
OPS_SRC     := $(filter-out src/ops/gpu_none.c,$(OPS_SRC))
ifeq ($(GPU),cuda)
GPU_SRC   := src/backend/gpu.cu $(wildcard src/ops/*.cu)
else
BACKEND_SRC += src/backend/gpu_none.c
OPS_SRC     += src/ops/gpu_none.c
GPU_SRC   :=
endif
LIB_SRC   := $(CORE_SRC) $(OPS_SRC) $(TEXT_SRC) $(MODEL_SRC) $(FAMILY_SRC) $(BACKEND_SRC)
# A kernel file shares its op's name (matmul.c, matmul.cu): .cu objects end in .cu.o.
LIB_OBJ   := $(LIB_SRC:%.c=$(BUILD)/obj/%.o) $(GPU_SRC:%.cu=$(BUILD)/obj/%.cu.o)

TOOL_SRC := $(wildcard tools/*.c)
TOOL_OBJ := $(TOOL_SRC:%.c=$(BUILD)/obj/%.o)
TESTS    := $(BUILD)/test_core $(BUILD)/test_model $(BUILD)/test_ops $(BUILD)/test_golden \
            $(BUILD)/test_forward $(BUILD)/test_text $(BUILD)/test_backend

.PHONY: all cpu cuda cuda-check hip test check-alloc golden clean FORCE
.SECONDARY:
.DEFAULT_GOAL := cpu

all: cpu
cpu: $(BUILD)/athanor

$(BUILD)/athanor: $(TOOL_OBJ) $(LIB_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_%: $(BUILD)/obj/tests/test_%.o $(LIB_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/obj/%.cu.o: %.cu | cuda-check
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# One ATH_FAMILY(name) line per family. Rewritten only when the list changes,
# so that a rebuild with the same FAMILIES does not recompile the registry.
$(FAMILIES_H): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' $(foreach f,$(FAMILIES),'ATH_FAMILY($(f))') > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv $@.tmp $@; fi

$(BUILD)/obj/src/model/registry.o: $(FAMILIES_H)

FORCE:

# AddressSanitizer reserves the address range the NVIDIA driver maps its
# memory into; the CUDA debug build lets the driver have it.
TEST_ENV := $(if $(and $(filter cuda,$(GPU)),$(filter 1,$(DEBUG))),ASAN_OPTIONS=protect_shadow_gap=0)

test: $(TESTS) $(BUILD)/athanor
	@set -e; for t in $(TESTS); do echo "== $$t"; $(TEST_ENV) ./$$t; done
	@set -e; if [ -d "$(ATH_MODEL_DIR)" ]; then \
		echo "== athanor inspect $(ATH_MODEL_DIR)"; \
		./$(BUILD)/athanor inspect "$(ATH_MODEL_DIR)" > /dev/null; echo "ok"; \
		echo "== athanor inspect $(ATH_MODEL_DIR) --roles"; \
		./$(BUILD)/athanor inspect "$(ATH_MODEL_DIR)" --roles > /dev/null; echo "ok"; \
	else echo "== athanor inspect: skipped, no $(ATH_MODEL_DIR)"; fi
ifneq ($(DEBUG),1)
	@if [ -d "$(ATH_MODEL_DIR)" ]; then $(MAKE) --no-print-directory check-alloc; fi
endif

# Fails if `athanor inspect` (with and without --roles) allocates heap memory
# outside the arenas.
check-alloc: $(BUILD)/athanor
	@echo "== heap allocations of athanor inspect"
	@sh tools/dev/check_alloc.sh $(BUILD)/athanor inspect "$(ATH_MODEL_DIR)"
	@echo "== heap allocations of athanor inspect --roles"
	@sh tools/dev/check_alloc.sh $(BUILD)/athanor inspect "$(ATH_MODEL_DIR)" --roles

# Regenerate the golden data from the pinned reference implementation
# (development only; needs the environment in tests/golden/requirements.txt).
golden:
	$(GOLDEN_PYTHON) tests/golden/dump.py "$(ATH_MODEL_DIR)" --out "$(ATH_GOLDEN_DIR)"
	$(GOLDEN_PYTHON) tests/golden/text.py "$(ATH_MODEL_DIR)" --out "$(ATH_GOLDEN_DIR)"

# `make cuda` builds everything with the CUDA backend; with `test` it also
# runs the suites of that build (GPU is set from the goals, above).
cuda: cuda-check $(BUILD)/athanor

cuda-check:
	@sh tools/dev/check_nvcc.sh "$(NVCC)" $(CUDA_ARCH)

hip:
	@echo "make hip: the HIP backend arrives in step 11 (docs/PLAN.md section 4)" >&2; exit 1

clean:
	rm -rf build

-include $(LIB_OBJ:.o=.d) $(TOOL_OBJ:.o=.d) $(TESTS:$(BUILD)/%=$(BUILD)/obj/tests/%.d)
