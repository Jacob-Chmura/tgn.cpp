FROM ubuntu:24.04

ARG CUDA_VERSION=cpu # Default is "cpu". Pass "12.6", "12.8", or "13.0" to trigger CUDA install.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    clang \
    libc++-dev \
    libc++abi-dev \
    libomp-18-dev \
    cmake \
    make \
    git \
    curl \
    wget \
    ca-certificates \
    && \
    if [ "$CUDA_VERSION" != "cpu" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        PACKAGE_SUFFIX=$(echo $CUDA_VERSION | sed 's/\./-/g') && \
        apt-get install -y cuda-toolkit-${PACKAGE_SUFFIX} ; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV CC=clang
ENV CXX=clang++
ENV PATH=${CUDA_VERSION:+/usr/local/cuda-${CUDA_VERSION}/bin:}${PATH}
ENV LD_LIBRARY_PATH=${CUDA_VERSION:+/usr/local/cuda-${CUDA_VERSION}/lib64:}${LD_LIBRARY_PATH}

WORKDIR /workspace
