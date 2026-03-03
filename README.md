## tgn.cpp

C++ Port of "Temporal Graph Networks for Deep Learning on Dynamic Graphs".

#### Prerequisites

```sh
# C++ Toolchain: Clang w/ C++20 and the LLVM STL
apt-get install -y clang libc++-dev libc++abi-dev

# TGUF Conversion Python Scripts use uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Usage

```sh
git clone git@github.com:Jacob-Chmura/tgn.cpp.git && cd tgn.cpp

# See available targets
make help

# Download `tgbl-wiki` data, convert to `.tguf` and run examples/link_pred.cpp.
make run-link-tgbl-wiki

# Download `tgbn-trade` data, convert to `.tguf` and run examples/node_pred.cpp
make run-node-tgbn-trade
```
