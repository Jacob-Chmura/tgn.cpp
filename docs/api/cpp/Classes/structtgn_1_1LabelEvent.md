---
title: tgn::LabelEvent
summary: Container for a label event at a single point in time.
---

# tgn::LabelEvent

Container for a label event at a single point in time.

`#include <tgn.h>`

## Public Attributes

|               | Name                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| torch::Tensor | **[n_id](Classes/structtgn_1_1LabelEvent.md#variable-n-id)** <br>Label Node Ids \[B\].               |
| torch::Tensor | **[target](Classes/structtgn_1_1LabelEvent.md#variable-target)** <br>Label targets \[B, label_dim\]. |

## Public Attributes Documentation

### variable n_id

```cpp
torch::Tensor n_id;
```

Label Node Ids \[B\].

### variable target

```cpp
torch::Tensor target;
```

Label targets \[B, label_dim\].

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
