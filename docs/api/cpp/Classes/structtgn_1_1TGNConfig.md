---
title: tgn::TGNConfig
summary: Configuration parameters for the TGN model architecture.
---

# tgn::TGNConfig

Configuration parameters for the TGN model architecture.

`#include <tgn.h>`

## Public Attributes

|             | Name                                                                                                                  |
| ----------- | --------------------------------------------------------------------------------------------------------------------- |
| std::size_t | **[embedding_dim](Classes/structtgn_1_1TGNConfig.md#variable-embedding-dim)** <br>TransformerConv embedding size.     |
| std::size_t | **[memory_dim](Classes/structtgn_1_1TGNConfig.md#variable-memory-dim)** <br>TGNMemory embedding size.                 |
| std::size_t | **[time_dim](Classes/structtgn_1_1TGNConfig.md#variable-time-dim)** <br>TimeEncoder embedding size.                   |
| std::size_t | **[num_heads](Classes/structtgn_1_1TGNConfig.md#variable-num-heads)** <br>TransformerConv multi-head attention heads. |
| std::size_t | **[num_nbrs](Classes/structtgn_1_1TGNConfig.md#variable-num-nbrs)** <br>RecencySampler neighbor buffer size.          |
| float       | **[dropout](Classes/structtgn_1_1TGNConfig.md#variable-dropout)** <br>TransformerConv dropout.                        |

## Public Attributes Documentation

### variable embedding_dim

```cpp
std::size_t embedding_dim = 100;
```

TransformerConv embedding size.

### variable memory_dim

```cpp
std::size_t memory_dim = 100;
```

TGNMemory embedding size.

### variable time_dim

```cpp
std::size_t time_dim = 100;
```

TimeEncoder embedding size.

### variable num_heads

```cpp
std::size_t num_heads = 2;
```

TransformerConv multi-head attention heads.

### variable num_nbrs

```cpp
std::size_t num_nbrs = 10;
```

RecencySampler neighbor buffer size.

### variable dropout

```cpp
float dropout = 0.1;
```

TransformerConv dropout.

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
