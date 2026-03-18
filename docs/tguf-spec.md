# Temporal Graph Unified Format

TGUF is a binary, flatbuffer-style on-disk format for temporal graph streams, designed for high-performance TGL workloads.

It enables zero-copy tensor access via memory mapping, allowing graph data to be loaded directly into frameworks like PyTorch without serialization overhead.
The format is optimized for sequential edge streams, making it well-suited for online temporal learning pipelines.

> **Note**: The binary layout is evolving and may not yet guarentee stability across versions.

## Key Features

- **Memory-mappable**: zero-copy reads into tensors
- **Compact contiguous disk-layout**: cache- and IO-efficient for sequential access patterns
- **Append-friendly**: efficient ingestion of temporal events
- **Static Node Feature support**: stored with direct indexing for random access
- **Pre-computed Negative Edges support**: negative destinations for link prediction
- **Dynamic Node Label Event support**: for node classification or regression
- **Pre-computed Data Split support**: time boundaries encoded in the file header

______________________________________________________________________

## TGUF Layout Overview

TGUF **single contiguous memory-mapped binary** with a fixed header and offset-based sections

| Section  | Field                  | Description                                                 | Required             | Bytes                                                              |
| -------- | ---------------------- | ----------------------------------------------------------- | -------------------- | ------------------------------------------------------------------ |
| `Header` | `magic`                | Format identification                                       | Yes (auto-generated) | sizeof(std::uint64_t)                                              |
| `Header` | `version`              | Format versioning                                           | Yes (auto-generated) | sizeof(std::uint64_t)                                              |
| `Header` | `edge_capacity`        | Max number of edges                                         | Yes                  | sizeof(std::uint64_t)                                              |
| `Header` | `label_capacity`       | Max number of label events                                  | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `node_capacity `       | Max number of nodes with static features                    | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `msg_dim `             | Fixed edge feature dimension                                | Yes                  | sizeof(std::uint64_t)                                              |
| `Header` | `label_dim`            | Fixed label target dimension                                | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `node_feat_dim `       | Fixed static node feature dimension                         | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `negatives_start_e_id` | For link prediction, e_id where pre-computed negative begin | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `negatives_per_edge`   | Fixed number of negative links per edge                     | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `val_start`            | Global edge index offset where validation split begins      | No                   | sizeof(std::uint64_t)                                              |
| `Header` | `test_start`           | Global edge index offset where test split begins            | No                   | sizeof(std::uint64_t)                                              |
| `Data`   | `src`                  | Source node IDs                                             | Yes                  | sizeof(std::uint64_t) * \|edge_capacity\|                          |
| `Data`   | `dst`                  | Destination node IDs                                        | Yes                  | sizeof(std::uint64_t) * \|edge_capacity\|                          |
| `Data`   | `time`                 | Edge Timestamps                                             | Yes                  | sizeof(std::uint64_t) * \|edge_capacity\|                          |
| `Data`   | `msg`                  | Edge Features                                               | Yes                  | sizeof(std::float32) * \|edge_capacity\| * \|msg_dim\|             |
| `Data`   | `neg_dst`              | Negative destinations                                       | No                   | sizeof(std::uint64_t) * \|edge_capacity\| * \|negatives_per_edge\| |
| `Data`   | `node_feat`            | Static Node Features                                        | No                   | sizeof(std::float32) * \|node_capacity\| * \|node_feat_dim\|       |
| `Data`   | `label_n_id`           | Node IDs for label events                                   | No                   | sizeof(std::uint64_t) * \|label_capacity\|                         |
| `Data`   | `label_time`           | Timestamps for label events                                 | No                   | sizeof(std::uint64_t) * \|label_capacity\|                         |
| `Data`   | `label_target`         | Label event targets                                         | No                   | sizeof(std::float32) * \|label_capacityy\| * \|label_dim\|         |
