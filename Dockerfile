FROM ubuntu:24.04

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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | BINDIR=/usr/local/bin sh

ENV CC=clang
ENV CXX=clang++

WORKDIR /workspace
