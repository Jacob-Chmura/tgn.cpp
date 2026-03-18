---
title: tgn::TGUFSchema
summary: metadata defining the layout of a Temporal Graph Unified Format file.
---

# tgn::TGUFSchema

metadata defining the layout of a Temporal Graph Unified Format file.

`#include <tgn.h>`

## Public Attributes

|                               | Name                                                                                                                                                                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| std::string                   | **[path](Classes/structtgn_1_1TGUFSchema.md#variable-path)** <br>Path to .tguf binary.                                                                                                                                                                                               |
| std::size_t                   | **[edge_capacity](Classes/structtgn_1_1TGUFSchema.md#variable-edge-capacity)** <br>Max number of edges.                                                                                                                                                                              |
| std::size_t                   | **[label_capacity](Classes/structtgn_1_1TGUFSchema.md#variable-label-capacity)** <br>Max number of label events.                                                                                                                                                                     |
| std::size_t                   | **[node_feat_capacity](Classes/structtgn_1_1TGUFSchema.md#variable-node-feat-capacity)** <br>Max nodes with static features.                                                                                                                                                         |
| std::size_t                   | **[msg_dim](Classes/structtgn_1_1TGUFSchema.md#variable-msg-dim)** <br>Fixed edge feature dimension.                                                                                                                                                                                 |
| std::size_t                   | **[label_dim](Classes/structtgn_1_1TGUFSchema.md#variable-label-dim)** <br>Fixed label target dimension.                                                                                                                                                                             |
| std::size_t                   | **[node_feat_dim](Classes/structtgn_1_1TGUFSchema.md#variable-node-feat-dim)** <br>Fixed static nod feature dimension.                                                                                                                                                               |
| std::size_t                   | **[negatives_start_e_id](Classes/structtgn_1_1TGUFSchema.md#variable-negatives-start-e-id)** <br>For link prediction evaluation, the e_id where pre-computed negatives begin.                                                                                                        |
| std::size_t                   | **[negatives_per_edge](Classes/structtgn_1_1TGUFSchema.md#variable-negatives-per-edge)** <br>Fixed number of negatives per edge.                                                                                                                                                     |
| std::optional\< std::size_t > | **[val_start](Classes/structtgn_1_1TGUFSchema.md#variable-val-start)** <br>Global index offset where the validation split begins. If `std::nullopt`, the dataset is treated as 100% training data unless overridden during [TGStore](Classes/classtgn_1_1TGStore.md) initialization. |
| std::optional\< std::size_t > | **[test_start](Classes/structtgn_1_1TGUFSchema.md#variable-test-start)** <br>Global index offset where the test split begins. Must be greater than or equal to [val_start](Classes/structtgn_1_1TGUFSchema.md#variable-val-start) if both are provided.                              |

## Public Attributes Documentation

### variable path

```cpp
std::string path;
```

Path to .tguf binary.

### variable edge_capacity

```cpp
std::size_t edge_capacity;
```

Max number of edges.

### variable label_capacity

```cpp
std::size_t label_capacity;
```

Max number of label events.

### variable node_feat_capacity

```cpp
std::size_t node_feat_capacity;
```

Max nodes with static features.

### variable msg_dim

```cpp
std::size_t msg_dim;
```

Fixed edge feature dimension.

### variable label_dim

```cpp
std::size_t label_dim;
```

Fixed label target dimension.

### variable node_feat_dim

```cpp
std::size_t node_feat_dim;
```

Fixed static nod feature dimension.

### variable negatives_start_e_id

```cpp
std::size_t negatives_start_e_id;
```

For link prediction evaluation, the e_id where pre-computed negatives begin.

### variable negatives_per_edge

```cpp
std::size_t negatives_per_edge;
```

Fixed number of negatives per edge.

### variable val_start

```cpp
std::optional< std::size_t > val_start = std::nullopt;
```

Global index offset where the validation split begins. If `std::nullopt`, the dataset is treated as 100% training data unless overridden during [TGStore](Classes/classtgn_1_1TGStore.md) initialization.

-

### variable test_start

```cpp
std::optional< std::size_t > test_start = std::nullopt;
```

Global index offset where the test split begins. Must be greater than or equal to [val_start](Classes/structtgn_1_1TGUFSchema.md#variable-val-start) if both are provided.

-

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
