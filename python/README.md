### tguf (Temporal Graph Universal Format)

Python bindings for the `tgn.cpp` TGUF builder.

### Prerequisites

Requires [uv](https://docs.astral.sh/uv/) and C++20 toolchain.

```sh
# Build an editable install with bindings from the project root
make python
```

### Quickstart

Our bindings use zero-copy semantics to ingest any Python object supporting the DLPack or Buffer protocol (e.g., numpy.ndarray, torch.Tensor).

```python
import tguf
import numpy as np
import torch

# Define your tguf schema
schema = tguf.TGUFSchema(
    path="my_data.tguf",
    edge_capacity=10,
    node_feat_capacity=5,
    node_feat_dim=4,
    msg_dim=8,
)
builder = tguf.TGUFBuilder(schema)

# Stream some edge events
edge_batch = tguf.Batch(
    src=np.array([0, 1], dtype=np.int64),
    dst=np.array([2, 3], dtype=np.int64),
    time=np.array([0, 1], dtype=np.int64),
    msg=np.random.randn(2, 8), dtype=np.float32),
)
builder.append_edges(batch)

# Stream through some node features
builder.append_node_feats(
    n_id=torch.arange(5, dtype=torch.int64) # tensor's work too
    node_feat = torch.randn(5, 4, dtype=torch.float32)
)

# Finalize the .tguf file and flush to disk
builder.finalize()
```
