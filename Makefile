CC      ?= cc
PY      ?= python

CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -O2 -ffp-contract=off
INCLUDES := -Iinclude
CFLAGS  += $(INCLUDES)

BUILD   := build
LIB     := $(BUILD)/libcinfer.a

LIB_SRCS := \
	src/arena.c \
	src/tensor.c \
	src/engine.c \
	src/layers/linear.c \
	src/layers/conv2d.c \
	src/layers/maxpool2d.c \
	src/layers/flatten.c \
	src/activations/relu.c \
	src/activations/sigmoid.c \
	src/activations/softmax.c

LIB_OBJS := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRCS))

TEST_NAMES := linear conv2d maxpool2d flatten relu sigmoid softmax end_to_end
TEST_SRCS  := $(foreach n,$(TEST_NAMES),tests/test_$(n).c)
TEST_BINS  := $(patsubst tests/%.c,$(BUILD)/%,$(TEST_SRCS))

NO_ALLOC_BIN := $(BUILD)/test_no_alloc_forward
FIXTURES := $(BUILD)/test_fixtures/end_to_end.cinf

.PHONY: all lib test demo export fixtures clean

all: lib test demo

lib: $(LIB)

$(LIB): $(LIB_OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

$(BUILD)/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

fixtures: $(FIXTURES)

$(BUILD)/test_fixtures/%.cinf: tests/gen_fixtures.py | $(LIB)
	@mkdir -p $(BUILD)/test_fixtures
	$(PY) tests/gen_fixtures.py

test: fixtures $(TEST_BINS) $(NO_ALLOC_BIN)
	@set -e; for t in $(TEST_BINS) $(NO_ALLOC_BIN); do \
		echo "==> $$t"; \
		$$t || { echo "FAIL: $$t"; exit 1; }; \
	done
	@echo "ALL TESTS PASSED"

$(BUILD)/test_%: tests/test_%.c $(LIB) | fixtures
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDES) -Itests $< -o $@ -L$(BUILD) -lcinfer -lm

$(NO_ALLOC_BIN): tests/test_no_alloc_forward.c $(LIB) | fixtures
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDES) -Itests $< -o $@ -L$(BUILD) -lcinfer -lm \
		-Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free

demo: fixtures
	@mkdir -p $(BUILD)
	$(PY) demos/mlp_demo.py
	$(MAKE) build/run_inference
	./build/run_inference build/mlp.cinf build/mlp.input.bin

build/run_inference: demos/run_inference.c $(LIB)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ -L$(BUILD) -lcinfer -lm

export: fixtures
	@mkdir -p $(BUILD)
	$(PY) demos/mlp_demo.py --export $(BUILD)/mlp.cinf --no-print

clean:
	rm -rf $(BUILD)
