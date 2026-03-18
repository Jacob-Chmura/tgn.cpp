---
title: tgn::TGUFBuilder
summary: High-performance writer for creating TGUF datasets on disk.
---

# tgn::TGUFBuilder

High-performance writer for creating TGUF datasets on disk.  [More...](#detailed-description)

`#include <tgn.h>`

## Public Functions

|      | Name                                                                                                                                                                                                                                   |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      | **[TGUFBuilder](Classes/classtgn_1_1TGUFBuilder.md#function-tgufbuilder)**(const [TGUFSchema](Classes/structtgn_1_1TGUFSchema.md) & schema)                                                                                            |
|      | **[~TGUFBuilder](Classes/classtgn_1_1TGUFBuilder.md#function-~tgufbuilder)**()                                                                                                                                                         |
| auto | **[append_edges](Classes/classtgn_1_1TGUFBuilder.md#function-append-edges)**(const [Batch](Classes/structtgn_1_1Batch.md) & batch) const<br>Appends a batch of edges to the persistent store.                                          |
| auto | **[append_labels](Classes/classtgn_1_1TGUFBuilder.md#function-append-labels)**(const torch::Tensor & n_id, const torch::Tensor & time, const torch::Tensor & target) const<br>Appends a batch of label events to the persistent store. |
| auto | **[append_node_feats](Classes/classtgn_1_1TGUFBuilder.md#function-append-node-feats)**(const torch::Tensor & n_id, const torch::Tensor & node_feat) const<br>Appends a batch of static node features to the persistent store.          |
| auto | **[finalize](Classes/classtgn_1_1TGUFBuilder.md#function-finalize)**()<br>Finalizes the .tguf file, writing headers and flushing buffers.                                                                                              |

## Detailed Description

```cpp
class tgn::TGUFBuilder;
```

High-performance writer for creating TGUF datasets on disk.

Uses an internal buffer strategy to minimize disk I/O.

## Public Functions Documentation

### function TGUFBuilder

```cpp
explicit TGUFBuilder(
    const TGUFSchema & schema
)
```

### function ~TGUFBuilder

```cpp
~TGUFBuilder()
```

### function append_edges

```cpp
auto append_edges(
    const Batch & batch
) const
```

Appends a batch of edges to the persistent store.

### function append_labels

```cpp
auto append_labels(
    const torch::Tensor & n_id,
    const torch::Tensor & time,
    const torch::Tensor & target
) const
```

Appends a batch of label events to the persistent store.

### function append_node_feats

```cpp
auto append_node_feats(
    const torch::Tensor & n_id,
    const torch::Tensor & node_feat
) const
```

Appends a batch of static node features to the persistent store.

### function finalize

```cpp
auto finalize()
```

Finalizes the .tguf file, writing headers and flushing buffers.

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
