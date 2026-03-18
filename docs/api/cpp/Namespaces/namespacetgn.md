---
title: tgn
summary: Temporal Graph Networks: A High-performance implementation.

---

# tgn

Temporal Graph Networks: A High-performance implementation.

## Classes

|        | Name                                                                                                                                |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| struct | **[tgn::Batch](Classes/structtgn_1_1Batch.md)** <br>Container for temporal edge data.                                               |
| struct | **[tgn::LabelEvent](Classes/structtgn_1_1LabelEvent.md)** <br>Container for a label event at a single point in time.                |
| struct | **[tgn::TGNConfig](Classes/structtgn_1_1TGNConfig.md)** <br>Configuration parameters for the TGN model architecture.                |
| class  | **[tgn::TGNImpl](Classes/classtgn_1_1TGNImpl.md)** <br>The core Temporal Graph Network module.                                      |
| class  | **[tgn::TGStore](Classes/classtgn_1_1TGStore.md)** <br>Abstract interface for temporal graph storage.                               |
| class  | **[tgn::TGUFBuilder](Classes/classtgn_1_1TGUFBuilder.md)** <br>High-performance writer for creating TGUF datasets on disk.          |
| struct | **[tgn::TGUFSchema](Classes/structtgn_1_1TGUFSchema.md)** <br>metadata defining the layout of a Temporal Graph Unified Format file. |

## Functions

|     | Name                                                                       |
| --- | -------------------------------------------------------------------------- |
|     | **[TORCH_MODULE](Namespaces/namespacetgn.md#function-torch-module)**(TGN ) |

## Functions Documentation

### function TORCH_MODULE

```cpp
TORCH_MODULE(
    TGN
)
```

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
