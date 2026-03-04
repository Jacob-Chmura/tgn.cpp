BUILD_DIR := build
CMAKE_FLAGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
NPROCS := $(shell nproc)
EXAMPLE_LINK := $(BUILD_DIR)/examples/tgn_link_pred
EXAMPLE_NODE := $(BUILD_DIR)/examples/tgn_node_pred

MAKEFLAGS += --no-print-directory
BUILD_TYPE ?= RelWithDebInfo

.PHONY: all
all: build

.PHONY: help
help:
	@echo "========================================================================"
	@echo " TGN Build System "
	@echo "========================================================================"
	@echo "Build Targets:"
	@echo "  make                     - Build project (current: $(BUILD_TYPE))"
	@echo "  make debug               - Build project with Debug symbols"
	@echo "  make release             - Build project with High optimization"
	@echo "  make tools               - Build CLI tools (tguf_cli)"
	@echo "  make examples            - Build tgn_link_prop and tgn_node_prop"
	@echo "  make clean               - Remove build directory"
	@echo ""
	@echo "Testing Targets:"
	@echo "  make test                - Run C++ unit tests (no Python dep)"
	@echo "  make test-integration    - Run Python-C++ TGUF roundtrip (requires uv)"
	@echo ""
	@echo "Run Targets (Download + TGUF Convert + Run):"
	@echo "  make run-link-<ds>       - Link prediction (e.g., make run-link-tgbl-wiki)"
	@echo "  make run-node-<ds>       - Node prediction (e.g., make run-node-tgbn-trade)"
	@echo ""
	@echo "Data Targets:"
	@echo "  make download-<ds>       - Download TGB dataset (e.g., make download-tgbl-wiki)"
	@echo "========================================================================"

$(BUILD_DIR)/CMakeCache.txt:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ..

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

.PHONY: tools
tools: config
	@cd $(BUILD_DIR) && cmake -DTGN_BUILD_TOOLS=ON ..
	@$(MAKE) build

.PHONY: examples
examples: config
	@cd $(BUILD_DIR) && cmake -DTGN_BUILD_EXAMPLES=ON ..
	@$(MAKE) build

.PHONY: test
test: config
	@cd $(BUILD_DIR) && cmake -DTGN_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
	$(MAKE) build
	@cd $(BUILD_DIR) && ctest -L unit --output-on-failure -j $(NPROCS)

.PHONY: test-integration
test-integration: tools
	@cd $(BUILD_DIR) && cmake -DTGN_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
	$(MAKE) build
	@cd $(BUILD_DIR) && ctest -L integration --output-on-failure -j $(NPROCS)

data/%.tguf: tools
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

.PHONY: download-%
download-%: data/%.tguf
	@echo "Dataset $* is up to date."

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
