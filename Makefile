# athanor build. Plain make, no configure step.
#
#   make            CPU build (default), output in build/
#   make test       build and run the test suites
#   make DEBUG=1    -O0 -g, ATH_ASSERT enabled, address and UB sanitizers
#   make cuda/hip   GPU builds (arrive in step 6 and step 11)
#   make clean

FAMILIES ?= qwen35
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

# Families are selected at compile time (docs/ARCHITECTURE.md, rule 1).
# Their sources appear in step 2; until then the list only has to name existing folders.
$(foreach f,$(FAMILIES),$(if $(wildcard src/models/$(f)/),,$(error unknown family '$(f)': no src/models/$(f)/)))
FAMILY_SRC := $(foreach f,$(FAMILIES),$(wildcard src/models/$(f)/*.c))

CORE_SRC := $(wildcard src/core/*.c)
LIB_SRC  := $(CORE_SRC) $(FAMILY_SRC)
LIB_OBJ  := $(LIB_SRC:%.c=$(BUILD)/obj/%.o)

TOOL_OBJ := $(BUILD)/obj/tools/athanor.o
TESTS    := $(BUILD)/test_core

.PHONY: all cpu cuda hip test clean
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

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

cuda:
	@echo "make cuda: the CUDA backend arrives in step 6 (docs/PLAN.md section 4)" >&2; exit 1

hip:
	@echo "make hip: the HIP backend arrives in step 11 (docs/PLAN.md section 4)" >&2; exit 1

clean:
	rm -rf build

-include $(LIB_OBJ:.o=.d) $(TOOL_OBJ:.o=.d) $(TESTS:$(BUILD)/%=$(BUILD)/obj/tests/%.d)
