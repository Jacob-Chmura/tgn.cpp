# tgn.cpp: Documentation

**tgn.cpp** is a library for large-scale Temporal Graph Learning, built around two components:

______________________________________________________________________

### 1. Temporal Graph Unified Format (TGUF)

A binary, flatbuffer-style on-disc format for graph streams, supporting:

- Dynamic node and edge events, static node features, pre-computed negatives (for link prediction)
- Zero-copy tensor reads via memory mapping for out-of-core training and inference
- Optimized sequential access patterns common in CTDG style methods

Includes a **storage engine API** for reading TGUF files.

> **Note**: We expose Python bindings for TGUF ingestion so that you can easily convert your own datasets into the binary file format. See the [Python API](./python-api.md) for more details.

See: [tguf-spec](./tguf-spec.md) for more details.

### 2. High-Performance TGN Implementation

A C++20 Port of [TGN](https://arxiv.org/abs/2006.10637) over pure LibTorch:

- Built on the TGUF storage engine
- Minimal abstractions, with efficient sampling kernels and data loading

______________________________________________________________________

## Out-of-the box examples

`tgn.cpp` includes a ready-to-run examples for [link prediction](../examples/link_pred.cpp) and [node prediction](../examples/node_pred.cpp).

These examples automatically:

- download and convert [TGB datasets](https://tgb.complexdatalab.com/) into TGUF format
- run the TGN model end-to-end
