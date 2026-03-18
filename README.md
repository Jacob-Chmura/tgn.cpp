<div align="center">

## Temporal Graph Learning on Streams that Exceed RAM

![C++20](https://img.shields.io/badge/C++-20-blue)
![Clang](https://img.shields.io/badge/Compiler-Clang-orange)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
[![Docs](https://img.shields.io/readthedocs/tgm?style=flat&label=Docs&labelColor=white&logo=readthedocs&logoColor=black)](https://tgn.cpp.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/Jacob-Chmura/tgn.cpp/ci.yml?label=Tests&style=flat&labelColor=white&logo=github-actions&logoColor=black)](https://github.com/Jacob-Chmura/tgn.cpp/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/Jacob-Chmura/tgn.cpp?style=flat&label=Coverage&labelColor=white&logo=codecov&logoColor=black)](https://codecov.io/gh/Jacob-Chmura/tgn.cpp)

</div>

`tgn.cpp` is a systems-first library for large-scale Temporal Graph Learning, built around two core components:

#### Temporal Graph Unified Format (TGUF)
A binary, flatbuffer-style memory mappable format for graph streams, supporting:

- Dynamic node and edge events, static node features, pre-computed negatives (for link prediction)
- Zero-copy tensor reads via memory mapping for out-of-core training and inference
- Optimized sequential access patterns common in CTDG style methods

#### High-Performance TGN Implementation
A C++20 Port of [TGN](https://arxiv.org/abs/2006.10637) over pure LibTorch:

- Built on the TGUF storage engine
- Minimal abstractions, with efficient sampling kernels and data loading

> \[!TIP\]
> Our [Python bindings](./python) for TGUF ingestion allow easy conversion of your dataset into TGUF

### Installation

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
> \[!Note\]
> Tested on Linux (Ubuntu 22.04+) and macOS (Apple Silicon)

```sh
git clone git@github.com:Jacob-Chmura/tgn.cpp.git && cd tgn.cpp

# See available targets
make help

# Download `tgbl-wiki` data, convert to `.tguf` and run examples/link_pred.cpp.
make run-link-tgbl-wiki

# Download `tgbn-trade` data, convert to `.tguf` and run examples/node_pred.cpp
make run-node-tgbn-trade
```
