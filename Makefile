BUILD_DIR := build
PROFILE_DIR := build-profile

CUDA_VERSION ?= cpu
GPU_ARCH ?= native

CMAKE_FLAGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_VERSION=$(CUDA_VERSION)

ifneq ($(CUDA_VERSION), cpu)
    CMAKE_FLAGS += -DCMAKE_CUDA_ARCHITECTURES=$(GPU_ARCH)

    # Add Torch-specific Arch list (converts 80 -> 8.0)
    ifneq ($(GPU_ARCH), native)
        TORCH_ARCH := $(shell echo $(GPU_ARCH) | sed 's/\([0-9]\)\([0-9]\)/\1.\2/')
        CMAKE_FLAGS += -DTORCH_CUDA_ARCH_LIST="$(TORCH_ARCH)"
    endif

    # Handle CUDA Compiler Path (Look in standard /usr/local/cuda-X.Y)
    CUDA_PATH := /usr/local/cuda-$(CUDA_VERSION)
    ifneq ("$(wildcard $(CUDA_PATH)/bin/nvcc)","")
        CMAKE_FLAGS += -DCMAKE_CUDA_COMPILER=$(CUDA_PATH)/bin/nvcc
    endif
endif

NPROCS := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)

EXAMPLE_LINK := $(BUILD_DIR)/examples/tgn_link_pred
EXAMPLE_NODE := $(BUILD_DIR)/examples/tgn_node_pred
EXAMPLE_LINK_PROF := $(PROFILE_DIR)/examples/tgn_link_pred
EXAMPLE_NODE_PROF := $(PROFILE_DIR)/examples/tgn_node_pred

MAKEFLAGS += --no-print-directory
BUILD_TYPE ?= Release

.PHONY: all
all: build

.PRECIOUS: data/%.tguf

.PHONY: help
help:
	@echo "========================================================================"
	@echo " TGN Build System "
	@echo "========================================================================"
	@echo "Build Targets:"
	@echo "  make                     - Build entire project (current: $(BUILD_TYPE))"
	@echo "  make tguf                - Build only the TGUF storage package"
	@echo "  make debug               - Build entire project with Debug symbols"
	@echo "  make release             - Build entire project with High optimization"
	@echo "  make examples            - Build tgn_link_prop and tgn_node_prop examples"
	@echo "  make clean               - Remove build directory"
	@echo ""
	@echo "Build Parameters (Optional):"
	@echo "  CUDA_VERSION=<ver>       - Build for CUDA (e.g., 12.6, 12.8, 13.0). Default: cpu"
	@echo "  GPU_ARCH=<arch>          - Compute capability (e.g., 80, 90, native). Default: native"
	@echo ""
	@echo "Documentation Targets:"
	@echo "  make docs                - Build project documentation"
	@echo "  make docs-serve          - Build and serve project documentation"
	@echo ""
	@echo "Testing Targets:"
	@echo "  make test                - Run all C++ unit tests (no Python dep)"
	@echo "  make test-tguf           - Run tguf C++ unit tests (no Python dep)"
	@echo "  make test-models         - Run tguf-models C++ unit tests (no Python dep)"
	@echo ""
	@echo "Run Targets (Download + TGUF Convert + Run):"
	@echo "  make run-link-<ds>       - Link prediction (e.g., make run-link-tgbl-wiki)"
	@echo "  make run-node-<ds>       - Node prediction (e.g., make run-node-tgbn-trade)"
	@echo ""
	@echo "Profiling Targets (Requires perf + sudo):"
	@echo "  make perf-link-<ds>       - Profile Link prediction (e.g., make perf-link-tgbl-wiki)"
	@echo "  make perf-node-<ds>       - Profile Node prediction (e.g., make perf-node-tgbn-trade)"
	@echo ""
	@echo "Data Targets:"
	@echo "  make download-<ds>       - Download TGB dataset (e.g., make download-tgbl-wiki)"
	@echo ""
	@echo "Python Targets:"
	@echo "  make python                - Build and install Python bindings"
	@echo "  make test-python           - Run Python-specific tests"
	@echo "  make clean-python          - Run Python-specific build artifacts"
	@echo "========================================================================"

$(BUILD_DIR)/CMakeCache.txt:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ..

$(PROFILE_DIR)/CMakeCache.txt:
	@mkdir -p $(PROFILE_DIR)
	@cd $(PROFILE_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTGN_BUILD_EXAMPLES=ON ..

.PHONY: config
config: $(BUILD_DIR)/CMakeCache.txt

.PHONY: build
build: config
	@cmake --build $(BUILD_DIR) --parallel $(NPROCS)

.PHONY: tguf
tguf: config
	@cmake --build $(BUILD_DIR) --target tguf --parallel $(NPROCS)

.PHONY: tguf-models
tguf-models: config
	@cmake --build $(BUILD_DIR) --target tguf_models --parallel $(NPROCS)

.PHONY: debug
debug:
	@$(MAKE) build BUILD_TYPE=Debug

.PHONY: release
release:
	@$(MAKE) build BUILD_TYPE=Release

.PHONY: examples
examples: config
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DTGN_BUILD_EXAMPLES=ON ..
	@cmake --build $(BUILD_DIR) --parallel $(NPROCS)

.PHONY: _test_config
_test_config:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=Debug -DTGN_BUILD_TESTS=ON ..

.PHONY: test
test: _test_config
	@cmake --build $(BUILD_DIR) --parallel $(NPROCS)
	@cd $(BUILD_DIR) && ctest --output-on-failure -j $(NPROCS)

.PHONY: test-tguf
test-tguf: _test_config
	@cmake --build $(BUILD_DIR) --target test_tguf --parallel $(NPROCS)
	@cd $(BUILD_DIR) && ctest -L tguf --output-on-failure -j $(NPROCS)

.PHONY: test-models
test-models: _test_config
	@cmake --build $(BUILD_DIR) --target test_tguf_models --parallel $(NPROCS)
	@cd $(BUILD_DIR) && ctest -L models --output-on-failure -j $(NPROCS)

data/%.tguf: python
	@mkdir -p data
	@if [ -f $@ ]; then \
		echo "Dataset $* already exists at $@, skipping download."; \
	else \
		./scripts/download_tgb_data.sh $*; \
	fi

.PHONY: run-link-%
run-link-%: examples data/%.tguf
	@$(EXAMPLE_LINK) data/$*.tguf

.PHONY: run-node-%
run-node-%: examples data/%.tguf
	@$(EXAMPLE_NODE) data/$*.tguf

.PHONY: profile-build
profile-build: $(PROFILE_DIR)/CMakeCache.txt
	@cmake --build $(PROFILE_DIR) --parallel $(NPROCS)

.PHONY: perf-link-%
perf-link-%: profile-build data/%.tguf
	@bash scripts/profile_tgn.sh $(EXAMPLE_LINK_PROF) data/$*.tguf

.PHONY: perf-node-%
perf-node-%: profile-build data/%.tguf
	@bash scripts/profile_tgn.sh $(EXAMPLE_NODE_PROF) data/$*.tguf

.PHONY: download-%
download-%: data/%.tguf
	@echo "Dataset $* is up to date."

.PHONY: python
python:
	@(cd python && \
		uv sync --group dev --no-install-project && \
		SKBUILD_CMAKE_ARGS="-DTGN_BUILD_PYTHON=ON" \
		uv pip install -e . --no-build-isolation)

.PHONY: test-python
test-python: python
	@(cd python && uv run pytest test/)

.PHONY: docs
docs: python
	@bash scripts/build_docs.sh

.PHONY: docs-serve
docs-serve: python
	@bash scripts/build_docs.sh serve

.PHONY: clean-python
clean-python:
	rm -rf python/build

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
