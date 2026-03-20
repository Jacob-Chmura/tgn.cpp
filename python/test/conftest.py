import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path
import tguf


@pytest.fixture
def tguf_path():
    with tempfile.NamedTemporaryFile(suffix=".tguf", delete=False) as tmp:
        path = Path(tmp.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def schema(tguf_path):
    return tguf.TGUFSchema(
        path=str(tguf_path),
        edge_capacity=100,
        node_feat_capacity=10,
        node_feat_dim=4,
        msg_dim=8,
        label_dim=1,
        label_capacity=3,
    )


@pytest.fixture
def get_data():
    def _get_data(lib, values, dtype):
        if lib == "torch":
            dtypes = {"int64": torch.long, "float32": torch.float32}
            return torch.tensor(values, dtype=dtypes[dtype])
        
        dtypes = {"int64": np.int64, "float32": np.float32}
        return np.array(values, dtype=dtypes[dtype])
    
    return _get_data
