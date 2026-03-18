---
title: tgn::TGNImpl
summary: The core Temporal Graph Network module.
---

# tgn::TGNImpl

The core Temporal Graph Network module.  [More...](#detailed-description)

`#include <tgn.h>`

Inherits from torch::nn::Module

## Public Functions

|                                     | Name                                                                                                                                                                                                                                                |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                     | **[TGNImpl](Classes/classtgn_1_1TGNImpl.md#function-tgnimpl)**(const [TGNConfig](Classes/structtgn_1_1TGNConfig.md) & cfg, const std::shared_ptr\< [TGStore](Classes/classtgn_1_1TGStore.md) > & store)                                             |
|                                     | **[~TGNImpl](Classes/classtgn_1_1TGNImpl.md#function-~tgnimpl)**()                                                                                                                                                                                  |
| auto                                | **[detach_memory](Classes/classtgn_1_1TGNImpl.md#function-detach-memory)**()<br>Detaches memory from the computational graph to truncate backprop (BPTT).                                                                                           |
| auto                                | **[reset_state](Classes/classtgn_1_1TGNImpl.md#function-reset-state)**()<br>Zeros out all node memory and resets last-update timestamps.                                                                                                            |
| auto                                | **[update_state](Classes/classtgn_1_1TGNImpl.md#function-update-state)**(const torch::Tensor & src, const torch::Tensor & dst, const torch::Tensor & time, const torch::Tensor & msg)<br>Updates internal memory given a batch of true edge events. |
| template \<typename... Ts> <br>auto | **[forward](Classes/classtgn_1_1TGNImpl.md#function-forward)**(const Ts &... inputs)<br>Variadic forward pass.                                                                                                                                      |

## Detailed Description

```cpp
class tgn::TGNImpl;
```

The core Temporal Graph Network module.

Manages node memory state and temporal neighborhood aggregation.

## Public Functions Documentation

### function TGNImpl

```cpp
TGNImpl(
    const TGNConfig & cfg,
    const std::shared_ptr< TGStore > & store
)
```

### function ~TGNImpl

```cpp
~TGNImpl()
```

### function detach_memory

```cpp
auto detach_memory()
```

Detaches memory from the computational graph to truncate backprop (BPTT).

### function reset_state

```cpp
auto reset_state()
```

Zeros out all node memory and resets last-update timestamps.

### function update_state

```cpp
auto update_state(
    const torch::Tensor & src,
    const torch::Tensor & dst,
    const torch::Tensor & time,
    const torch::Tensor & msg
)
```

Updates internal memory given a batch of true edge events.

### function forward

```cpp
template <typename... Ts>
inline auto forward(
    const Ts &... inputs
)
```

Variadic forward pass.

**Parameters**:

- **inputs** Tensors of node IDs to compute embeddings for.

**Return**: A tuple of embeddings \[B, embedding_dim\] in same order as inputs.

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
