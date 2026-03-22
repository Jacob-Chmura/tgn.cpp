<div align="center">

## Temporal Graph Learning on Streams that Exceed RAM

![C++20](https://img.shields.io/badge/C++-20-blue?style=flat&labelColor=white&logo=c%2B%2B&logoColor=black)
![Clang](https://img.shields.io/badge/Compiler-Clang-orange?style=flat&labelColor=white&logo=clang&logoColor=black)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat&labelColor=white&logo=nvidia&logoColor=76B900)
![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat&labelColor=white&logo=nvidia&logoColor=76B900)
![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat&labelColor=white&logo=nvidia&logoColor=76B900)
[![Docs](https://img.shields.io/readthedocs/tgncpp?style=flat&label=Docs&labelColor=white&logo=readthedocs&logoColor=black)](https://tgncpp.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/Jacob-Chmura/tgn.cpp/ci.yml?label=Tests&style=flat&labelColor=white&logo=github-actions&logoColor=black)](https://github.com/Jacob-Chmura/tgn.cpp/actions/workflows/ci.yml)

</div>

**tgn.cpp** is a library for large-scale Temporal Graph Learning, built around two components:

#### 1. Temporal Graph Unified Format (TGUF)

A binary, flatbuffer-style on-disc format for graph streams, supporting:

- Dynamic node and edge events, static node features, pre-computed negatives (for link prediction)
- Zero-copy tensor reads via memory mapping for out-of-core training and inference
- Optimized sequential access patterns common in CTDG style methods

#### 2. High-Performance TGN Implementation

A C++20 Port of [TGN](https://arxiv.org/abs/2006.10637) over pure LibTorch:

- Built on the TGUF storage engine
- Minimal abstractions, with efficient sampling kernels and data loading

> \[!TIP\]
> Use the [Python bindings](./python) for easy conversion of your datasets into TGUF

### Installation

You should just use the [Dockerfile](./Dockerfile):

```sh
# Build for CPU (default)
docker build -t tgn-dev:cpu .

# Build for specific CUDA drivers (e.g. 12.6 for A100/H100)
docker build --build-arg CUDA_VERSION=12.6 -t tgn-dev:cu126 .
```

If you prefer a bare-metal install:

##### Linux

```sh
# C++ Toolchain: Clang w/ C++20 and the LLVM STL
sudo apt-get install -y clang libc++-dev libc++abi-dev
```

If you want to run with CUDA support, refer to [nvidia docs](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local) for installation steps.

##### MacOS

```sh
# CMake and OpenMP runtime
brew install cmake libomp
```

> \[!Important\]
> **Platform Support**:

> | Variable     | Description                                  | Options                       | Linux | macOS | Default  |
> | ------------ | -------------------------------------------- | ----------------------------- | ------------------------- | -------------------------- | -------- |
> | CUDA_VERSION | CUDA backend | `cpu`, `12.6`, `12.8`, `13.0` | All | `cpu` (Apple Silicon) | `cpu`    |
> | GPU_ARCH     | Compute Capability (GPU architecture) | `80`, `90`, `native`          | All | N/A | `native` |

##### TGUF Conversion Scripts use [uv](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Usage

#### Setup

```sh
# Clone the repo
git clone git@github.com:Jacob-Chmura/tgn.cpp.git && cd tgn.cpp

# See all available targets
make help
```

#### Running on CPU

```sh

# Download `tgbl-wiki` data, convert to `.tguf` and run examples/link_pred.cpp.
make run-link-tgbl-wiki

# Download `tgbn-trade` data, convert to `.tguf` and run examples/node_pred.cpp
make run-node-tgbn-trade
```

#### Running on GPU (Linux only)

```sh
# Example: Cuda 12.6 on an A100 (Arch 80)
CUDA_VERSION=12.6 GPU_ARCH=80 make run-link-tgbl-wiki
```

> \[!TIP\]
> Use `nvidia-smi` to check your *CUDA_VERSION* and *GPU_ARCH*
