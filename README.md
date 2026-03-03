## tgn.cpp

C++ Port of "Temporal Graph Networks for Deep Learning on Dynamic Graphs".

### Prerequisites

**C++ Toolchain**:

```sh
# Clang w/ C++20 and the LLVM STL
apt-get install -y clang libc++-dev libc++abi-dev
```

**Python** [TGUF conversion scripts](./tools/) use [uv](https://docs.astral.sh/uv/) to manage dependencies:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Usage

```sh
git clone git@github.com:Jacob-Chmura/tgn.cpp.git
cd tgn.cpp
make help # See available build/execution targets
```

#### Link Prediction

Downloads `tgbl-wiki` data, converts to `.tguf` and runs the [link prediction example](./examples/link_pred.cpp)

```
make run-link-tgbl-wiki
```

#### Node Prediction

Downloads `tgbn-trade` data, converts to `.tguf` and runs the [node prediction example](./examples/node_pred.cpp)

```
make run-node-tgbn-trade
```
