BUILD_DIR := build
PROFILE_DIR := build-profile
CMAKE_FLAGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
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
	@echo "  make                     - Build project (current: $(BUILD_TYPE))"
	@echo "  make debug               - Build project with Debug symbols"
	@echo "  make release             - Build project with High optimization"
	@echo "  make examples            - Build tgn_link_prop and tgn_node_prop"
	@echo "  make clean               - Remove build directory"
	@echo ""
	@echo "Documentation Targets:"
	@echo "  make docs                - Build project documentation"
	@echo "  make docs-serve          - Build and serve project documentation"
	@echo ""
	@echo "Testing Targets:"
	@echo "  make test                - Run C++ unit tests (no Python dep)"
	@echo "  make test-integration    - Run Python-C++ TGUF roundtrip (requires uv)"
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
	@$(MAKE) build

.PHONY: test
test:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=Debug -DTGN_BUILD_TESTS=ON ..
	@$(MAKE) build BUILD_TYPE=Debug
	@cd $(BUILD_DIR) && ctest -L unit --output-on-failure -j $(NPROCS)

.PHONY: test-integration
test-integration: python
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=Debug -DTGN_BUILD_TESTS=ON ..
	@$(MAKE) build BUILD_TYPE=Debug
	@cd $(BUILD_DIR) && ctest -L integration --output-on-failure -j $(NPROCS)

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
