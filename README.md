## tgn.cpp

C++ Port of "Temporal Graph Networks for Deep Learning on Dynamic Graphs".

> \[!Note\]
> Tested on Linux (Ubuntu 22.04+) and macOS (Apple Silicon)

### Prerequisites

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

### Python Bindings

We expose [Python bindings](./python) for the `TGUFBuilder` via [nanobind](https://nanobind.readthedocs.io/en/latest/).
