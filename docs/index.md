# tgn.cpp: Documentation

`tgn.cpp` is a systems-first library for **Temporal Graph Learning at scale**. It is build around two core components:

______________________________________________________________________

### 1. Temporal Graph Unified Format (TGUF)

A custom binary, flatbuffer-style memory-mappable file format for temporal graph streams:

- Stores edge streams, static node features, dynamic node labels, negative edges, data splits
- Enables zero-copy tensor reads via memory mapping
- Designed to support out-of-core training and inference
- Optimized for sequential temporal access patterns

Includes a **storage engine API** for reading TGUF files.

> **Note**: We expose Python bindings for TGUF ingestion so that you can easily convert your own datasets into the binary file format. See the [Python API](./python-api.md) for more details.

See: [tguf-spec](./tguf-spec.md) for more details.

### 2. High-Performance TGN Implementation

A systems-optimized C++20 implementation of [TGN](https://arxiv.org/abs/2006.10637) over pure LibTorch:

- Built directly on top of the TGUF storage engine
- Efficient temporal sampling and data loading kernels
- Minimal abstractions for predictable performance

______________________________________________________________________

## Out-of-the box examples

`tgn.cpp` includes a ready-to-run examples for [link prediction](../examples/link_pred.cpp) and [node prediction](../examples/node_pred.cpp).

These examples automatically:

- download and convert [TGB datasets](https://tgb.complexdatalab.com/) into TGUF format
- run the TGN model end-to-end
