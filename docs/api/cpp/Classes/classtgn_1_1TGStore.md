---
title: tgn::TGStore
summary: Abstract interface for temporal graph storage.
---

# tgn::TGStore

Abstract interface for temporal graph storage.  [More...](#detailed-description)

`#include <tgn.h>`

## Public Classes

|        | Name                                                                                                                        |
| ------ | --------------------------------------------------------------------------------------------------------------------------- |
| struct | **[IndexRange](Classes/structtgn_1_1TGStore_1_1IndexRange.md)** <br>A contiguous slice of the graph (e.g., training split). |

## Public Types

|            | Name                                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| enum class | **[NegStrategy](Classes/classtgn_1_1TGStore.md#enum-negstrategy)** { None, Random, PreComputed}<br>Determines how negative samples are generated during [get_batch()](<>). |

## Public Functions

|              | Name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| virtual      | **[~TGStore](Classes/classtgn_1_1TGStore.md#function-~tgstore)**() =default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| virtual auto | **[edge_count](Classes/classtgn_1_1TGStore.md#function-edge-count)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| virtual auto | **[node_count](Classes/classtgn_1_1TGStore.md#function-node-count)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| virtual auto | **[msg_dim](Classes/classtgn_1_1TGStore.md#function-msg-dim)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| virtual auto | **[label_dim](Classes/classtgn_1_1TGStore.md#function-label-dim)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| virtual auto | **[node_feat_dim](Classes/classtgn_1_1TGStore.md#function-node-feat-dim)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| virtual auto | **[train_split](Classes/classtgn_1_1TGStore.md#function-train-split)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| virtual auto | **[val_split](Classes/classtgn_1_1TGStore.md#function-val-split)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| virtual auto | **[test_split](Classes/classtgn_1_1TGStore.md#function-test-split)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| virtual auto | **[train_label_split](Classes/classtgn_1_1TGStore.md#function-train-label-split)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| virtual auto | **[val_label_split](Classes/classtgn_1_1TGStore.md#function-val-label-split)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| virtual auto | **[test_label_split](Classes/classtgn_1_1TGStore.md#function-test-label-split)**() const =0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| virtual auto | **[get_batch](Classes/classtgn_1_1TGStore.md#function-get-batch)**(std::size_t start, std::size_t size, [NegStrategy](Classes/classtgn_1_1TGStore.md#enum-negstrategy) strategy =[NegStrategy::None](Classes/classtgn_1_1TGStore.md#enumvalue-none)) const =0<br>Retrieves a zero-copy slice of the graph.                                                                                                                                                                                                                                                                        |
| virtual auto | **[gather_timestamps](Classes/classtgn_1_1TGStore.md#function-gather-timestamps)**(const torch::Tensor & e_id) const =0<br>Performs a vectorized random-access gather of edge timestamps.                                                                                                                                                                                                                                                                                                                                                                                         |
| virtual auto | **[gather_msgs](Classes/classtgn_1_1TGStore.md#function-gather-msgs)**(const torch::Tensor & e_id) const =0<br>Performs a vectorized random-access gather of edge messages.                                                                                                                                                                                                                                                                                                                                                                                                       |
| virtual auto | **[gather_node_feats](Classes/classtgn_1_1TGStore.md#function-gather-node-feats)**(const torch::Tensor & n_id) const =0<br>Performs a vectorized random-access gather of node features.                                                                                                                                                                                                                                                                                                                                                                                           |
| virtual auto | **[get_edge_cutoff_for_label_event](Classes/classtgn_1_1TGStore.md#function-get-edge-cutoff-for-label-event)**(std::size_t l_id) const =0<br>Retrieves the maximum edge_id that can be safely processed before a label.                                                                                                                                                                                                                                                                                                                                                           |
| virtual auto | **[get_label_event](Classes/classtgn_1_1TGStore.md#function-get-label-event)**(std::size_t l_id) const =0<br>Retrieves the metadata and target for a specific label event.                                                                                                                                                                                                                                                                                                                                                                                                        |
| auto         | **[from_memory](Classes/classtgn_1_1TGStore.md#function-from-memory)**(const [Batch](Classes/structtgn_1_1Batch.md) & edges, const std::optional\< torch::Tensor > & node_feats =std::nullopt, const std::optional\< torch::Tensor > & label_n_id =std::nullopt, const std::optional\< torch::Tensor > & label_time =std::nullopt, const std::optional\< torch::Tensor > & label_target =std::nullopt, std::optional\< std::size_t > val_start =std::nullopt, std::optional\< std::size_t > test_start =std::nullopt)<br>Factory method for a high-speed, purely RAM-based store. |
| auto         | **[from_tguf](Classes/classtgn_1_1TGStore.md#function-from-tguf)**(const std::string & path, std::optional\< std::size_t > val_start =std::nullopt, std::optional\< std::size_t > test_start =std::nullopt)<br>Factory method for memory-mapped storage from a TGUF file.                                                                                                                                                                                                                                                                                                         |

## Detailed Description

```cpp
class tgn::TGStore;
```

Abstract interface for temporal graph storage.

Implementations can be purely in-memory or memory-mapped TGUF files.

## Public Types Documentation

### enum NegStrategy

| Enumerator  | Value | Description                                         |
| ----------- | ----- | --------------------------------------------------- |
| None        |       | No negatives (inference or node-level tasks).       |
| Random      |       | Samples one random negative node per edge.          |
| PreComputed |       | Uses the fixed negatives stored in TGUF (for eval). |

Determines how negative samples are generated during [get_batch()](<>).

## Public Functions Documentation

### function ~TGStore

```cpp
virtual ~TGStore() =default
```

### function edge_count

```cpp
virtual auto edge_count() const =0
```

### function node_count

```cpp
virtual auto node_count() const =0
```

### function msg_dim

```cpp
virtual auto msg_dim() const =0
```

### function label_dim

```cpp
virtual auto label_dim() const =0
```

### function node_feat_dim

```cpp
virtual auto node_feat_dim() const =0
```

### function train_split

```cpp
virtual auto train_split() const =0
```

### function val_split

```cpp
virtual auto val_split() const =0
```

### function test_split

```cpp
virtual auto test_split() const =0
```

### function train_label_split

```cpp
virtual auto train_label_split() const =0
```

### function val_label_split

```cpp
virtual auto val_label_split() const =0
```

### function test_label_split

```cpp
virtual auto test_label_split() const =0
```

### function get_batch

```cpp
virtual auto get_batch(
    std::size_t start,
    std::size_t size,
    NegStrategy strategy =NegStrategy::None
) const =0
```

Retrieves a zero-copy slice of the graph.

**Parameters**:

- **start** The starting edge ID.

- **size** The number of edges to include.

- **strategy** The negative sampling strategy to apply.

-

### function gather_timestamps

```cpp
virtual auto gather_timestamps(
    const torch::Tensor & e_id
) const =0
```

Performs a vectorized random-access gather of edge timestamps.

**Parameters**:

- **e_id** Tensor of edge indices \[num_edges\].

**Return**: torch::Tensor of timestamps \[num_edges\].

**Note**: Optimized for memory-mapped I/O; performance may vary based on disk locality.

-

### function gather_msgs

```cpp
virtual auto gather_msgs(
    const torch::Tensor & e_id
) const =0
```

Performs a vectorized random-access gather of edge messages.

**Parameters**:

- **e_id** Tensor of edge indices \[num_edges\].

**Return**: torch::Tensor of messages \[num_edges, msg_dim\].

-

### function gather_node_feats

```cpp
virtual auto gather_node_feats(
    const torch::Tensor & n_id
) const =0
```

Performs a vectorized random-access gather of node features.

**Parameters**:

- **n_id** Tensor of node indices \[num_nodes\].

**Return**: torch::Tensor of features \[num_nodes, node_feat_dim\].

-

### function get_edge_cutoff_for_label_event

```cpp
virtual auto get_edge_cutoff_for_label_event(
    std::size_t l_id
) const =0
```

Retrieves the maximum edge_id that can be safely processed before a label.

\*\* To prevent information leakage (look-ahead bias), the model state should only be updated with edges occurring before the timestamp of the label event `l_id`.

- l_idThe index of the label event.

The upper-bound edge_id (exclusive) for model state updates.

### function get_label_event

```cpp
virtual auto get_label_event(
    std::size_t l_id
) const =0
```

Retrieves the metadata and target for a specific label event.

**Parameters**:

- **l_id** The index of the label event.

**Return**: A [LabelEvent](Classes/structtgn_1_1LabelEvent.md) containing affected node IDs and target values.

-

### function from_memory

```cpp
static auto from_memory(
    const Batch & edges,
    const std::optional< torch::Tensor > & node_feats =std::nullopt,
    const std::optional< torch::Tensor > & label_n_id =std::nullopt,
    const std::optional< torch::Tensor > & label_time =std::nullopt,
    const std::optional< torch::Tensor > & label_target =std::nullopt,
    std::optional< std::size_t > val_start =std::nullopt,
    std::optional< std::size_t > test_start =std::nullopt
)
```

Factory method for a high-speed, purely RAM-based store.

### function from_tguf

```cpp
static auto from_tguf(
    const std::string & path,
    std::optional< std::size_t > val_start =std::nullopt,
    std::optional< std::size_t > test_start =std::nullopt
)
```

Factory method for memory-mapped storage from a TGUF file.

Supports datasets larger than available system RAM.

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
