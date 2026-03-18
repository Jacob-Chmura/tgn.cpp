<div align="center">

<h3 style="font-size: 28px">Temporal Graph Learning on Graphs that Exceed RAM </h3>
<a href="https://tgn.cpp.readthedocs.io/en/latest"/><strong style="font-size: 18px;"/>Read Our Docs»</strong></a>

</div>

**tgn.cpp** is built around two core components:

**1. Temporal Graph Unified Format (TGUF)**: A binary, flatbuffer-style memory mappable format for graph streams, supporting:

- Dynamic node/edge events, static node features
- Pre-computed negatives (for link prediction)
- Zero-copy tensor reads via memory mapping
- Out-of-core training and inference
- Optimized sequential access patterns common in CTDG style methods

**2. High-Performance TGN Implementation**: A C++20 Port of [TGN](https://arxiv.org/abs/2006.10637) over pure LibTorch:

- Built on the TGUF storage engine
- Minimal abstractions, with efficient sampling kernels and data loading

> \[!TIP\]
> Our [Python bindings](./python) for TGUF ingestion allow easy conversion of your dataset into TGUF

### Prerequisites

> \[!Note\]
> Tested on Linux (Ubuntu 22.04+) and macOS (Apple Silicon)

You should just use the [Dockerfile](./Dockerfile), but if you prefer to install dependencies manually, see below.

##### Linux

```sh
# C++ Toolchain: Clang w/ C++20 and the LLVM STL
sudo apt-get install -y clang libc++-dev libc++abi-dev
```

##### MacOS

```sh
# OpenMP runtime
brew install libomp
```

##### TGUF Conversion Scripts use [uv](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Usage

```sh
git clone git@github.com:Jacob-Chmura/tgn.cpp.git && cd tgn.cpp

# See available targets
make help

# Download `tgbl-wiki` data, convert to `.tguf` and run examples/link_pred.cpp.
make run-link-tgbl-wiki

# Download `tgbn-trade` data, convert to `.tguf` and run examples/node_pred.cpp
make run-node-tgbn-trade
```
