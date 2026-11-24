# athanor build. Plain make, no configure step.
#
#   make            CPU build (default), output in build/
#   make test       build and run the test suites
#   make check-alloc  verify that inspect allocates only through arenas
#   make golden     regenerate tests/golden/data from the reference implementation
#   make DEBUG=1    -O0 -g, ATH_ASSERT enabled, address and UB sanitizers
#   make cuda/hip   GPU builds (arrive in step 6 and step 11)
#   make clean

FAMILIES ?= qwen35
# Checkpoint and golden data used by the tests that need them (skipped if absent).
ATH_MODEL_DIR  ?= models/Qwen3.5-0.8B
ATH_GOLDEN_DIR ?= tests/golden/data/qwen35-0.8b
export ATH_MODEL_DIR ATH_GOLDEN_DIR
# Python of the golden-data environment (tests/golden/README.md).
GOLDEN_PYTHON ?= tests/golden/.venv/bin/python
CC       ?= cc
ifeq ($(DEBUG),1)
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
LIB_SRC   := $(CORE_SRC) $(OPS_SRC) $(TEXT_SRC) $(MODEL_SRC) $(FAMILY_SRC)
LIB_OBJ   := $(LIB_SRC:%.c=$(BUILD)/obj/%.o)

TOOL_SRC := $(wildcard tools/*.c)
TOOL_OBJ := $(TOOL_SRC:%.c=$(BUILD)/obj/%.o)
TESTS    := $(BUILD)/test_core $(BUILD)/test_model $(BUILD)/test_ops $(BUILD)/test_golden \
            $(BUILD)/test_forward $(BUILD)/test_text

.PHONY: all cpu cuda hip test check-alloc golden clean FORCE
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

# One ATH_FAMILY(name) line per family. Rewritten only when the list changes,
# so that a rebuild with the same FAMILIES does not recompile the registry.
$(FAMILIES_H): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' $(foreach f,$(FAMILIES),'ATH_FAMILY($(f))') > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv $@.tmp $@; fi

$(BUILD)/obj/src/model/registry.o: $(FAMILIES_H)

FORCE:

test: $(TESTS) $(BUILD)/athanor
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done
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

cuda:
	@echo "make cuda: the CUDA backend arrives in step 6 (docs/PLAN.md section 4)" >&2; exit 1

hip:
	@echo "make hip: the HIP backend arrives in step 11 (docs/PLAN.md section 4)" >&2; exit 1

clean:
	rm -rf build

-include $(LIB_OBJ:.o=.d) $(TOOL_OBJ:.o=.d) $(TESTS:$(BUILD)/%=$(BUILD)/obj/tests/%.d)
