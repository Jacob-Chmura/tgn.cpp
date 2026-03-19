FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y \
    clang \
    libc++-dev \
    libc++abi-dev \
    libomp-18-dev \
    cmake \
    make \
    git \
    curl \
    ca-certificates \
    cuda-toolkit-12-6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV CC=clang
ENV CXX=clang++

WORKDIR /workspace
