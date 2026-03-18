---
title: tgn::Batch
summary: Container for temporal edge data.
---

# tgn::Batch

Container for temporal edge data.

`#include <tgn.h>`

## Public Attributes

|                                 | Name                                                                                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| torch::Tensor                   | **[src](Classes/structtgn_1_1Batch.md#variable-src)** <br>Source node IDs \[B\].                                                                |
| torch::Tensor                   | **[dst](Classes/structtgn_1_1Batch.md#variable-dst)** <br>Destination node IDs \[B\].                                                           |
| torch::Tensor                   | **[time](Classes/structtgn_1_1Batch.md#variable-time)** <br>Timestamps \[B\].                                                                   |
| torch::Tensor                   | **[msg](Classes/structtgn_1_1Batch.md#variable-msg)** <br>Edge features \[B, msg_dim\].                                                         |
| std::optional\< torch::Tensor > | **[neg_dst](Classes/structtgn_1_1Batch.md#variable-neg-dst)** <br>Optional negative destinations for link prediction \[B, negatives_per_edge\]. |

## Public Attributes Documentation

### variable src

```cpp
torch::Tensor src;
```

Source node IDs \[B\].

### variable dst

```cpp
torch::Tensor dst;
```

Destination node IDs \[B\].

### variable time

```cpp
torch::Tensor time;
```

Timestamps \[B\].

### variable msg

```cpp
torch::Tensor msg;
```

Edge features \[B, msg_dim\].

### variable neg_dst

```cpp
std::optional< torch::Tensor > neg_dst;
```

Optional negative destinations for link prediction \[B, negatives_per_edge\].

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
