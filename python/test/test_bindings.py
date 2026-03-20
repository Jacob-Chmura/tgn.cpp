import tempfile
from pathlib import Path

import numpy as np
import pytest
import tguf
import torch


def get_data(lib, values, dtype):
    if lib == "torch":
        dtypes = {"int64": torch.long, "float32": torch.float32}
        return torch.tensor(values, dtype=dtypes[dtype])

    dtypes = {"int64": np.int64, "float32": np.float32}
    return np.array(values, dtype=dtypes[dtype])


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
        val_start=8,
        test_start=9,
    )


def test_index_range():
    ir = tguf.IndexRange(10, 50)
    assert ir.start == 10
    assert ir.end == 50
    assert ir.size == 40


def test_neg_strategy_exists():
    assert hasattr(tguf.NegStrategy, "None_")
    assert hasattr(tguf.NegStrategy, "Random")
    assert hasattr(tguf.NegStrategy, "PreComputed")


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_batch_init(lib):
    tguf.Batch(
        src=get_data(lib, [0, 1, 2], "int64"),
        dst=get_data(lib, [3, 4, 5], "int64"),
        time=get_data(lib, [10, 11, 12], "int64"),
        msg=get_data(lib, np.random.randn(3, 8), "float32"),
    )


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_batch_init_with_negs(lib):
    tguf.Batch(
        src=get_data(lib, [0, 1, 2], "int64"),
        dst=get_data(lib, [3, 4, 5], "int64"),
        time=get_data(lib, [10, 11, 12], "int64"),
        msg=get_data(lib, np.random.randn(3, 8), "float32"),
        neg_dst=get_data(lib, [3, 4, 5], "int64"),
    )


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_label_event_init(lib):
    tguf.LabelEvent(
        n_id=get_data(lib, [0, 1, 2], "int64"),
        target=get_data(lib, np.random.randn(3, 8), "float32"),
    )


def test_schema_init():
    schema = tguf.TGUFSchema(path="test.tguf", edge_capacity=500, msg_dim=16)

    assert schema.edge_capacity == 500
    assert schema.label_dim == 0
    assert schema.val_start is None
    assert schema.test_start is None

    schema.val_start = 1000
    assert schema.val_start == 1000


def test_tgstore_from_memory():
    num_edges = 20
    batch = tguf.Batch(
        src=get_data("numpy", np.arange(num_edges), "int64"),
        dst=get_data("numpy", np.arange(num_edges) + 1, "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.random.randn(num_edges, 8), "float32"),
    )

    store = tguf.TGStore.from_memory(edges=batch, val_start=15, test_start=18)

    assert store.edge_count == num_edges
    assert store.train_split.end == 15
    assert store.val_split.size == 3
    assert store.test_split.start == 18

    b = store.get_batch(0, 5, tguf.NegStrategy.None_)
    assert b.src.shape[0] == 5
    assert torch.is_tensor(b.src)
    assert b.dst.shape[0] == 5
    assert torch.is_tensor(b.dst)
    assert b.time.shape[0] == 5
    assert torch.is_tensor(b.time)
    assert b.msg.shape[0] == 5
    assert torch.is_tensor(b.msg)

    assert b.neg_dst is None


def test_tgstore_from_tguf(schema):
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=get_data("numpy", np.arange(10), "int64"),
        dst=get_data("numpy", np.arange(10) + 1, "int64"),
        time=get_data("numpy", np.arange(10) * 10, "int64"),
        msg=get_data("numpy", np.random.randn(10, 8), "float32"),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    assert store.edge_count == 10
    assert store.msg_dim == 8
    # TODO(kuba)
    # e_id = torch.tensor([0, 5, 9], dtype=torch.long)
    # t = store.gather_timestamps(e_id)
    # assert t.shape[0] == 3
    # assert t[0] == 0
    # assert t[1] == 50


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_builder_append_edges(schema, lib, tguf_path):
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=get_data(lib, [0, 1], "int64"),
        dst=get_data(lib, [1, 0], "int64"),
        time=get_data(lib, [10, 11], "int64"),
        msg=get_data(lib, np.random.randn(2, 8), "float32"),
    )
    builder.append_edges(batch)
    builder.finalize()

    assert tguf_path.exists()
    assert tguf_path.stat().st_size > 0


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_builder_append_labels(schema, lib, tguf_path):
    builder = tguf.TGUFBuilder(schema)

    builder.append_labels(
        get_data(lib, [0, 1], "int64"),
        get_data(lib, [10, 11], "int64"),
        get_data(lib, np.random.randn(2, 1), "float32"),
    )

    builder.finalize()
    assert tguf_path.exists()
    assert tguf_path.stat().st_size > 0


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_builder_append_node_feats(schema, lib, tguf_path):
    builder = tguf.TGUFBuilder(schema)

    builder.append_node_feats(
        get_data(lib, [0, 1], "int64"), get_data(lib, np.random.randn(2, 4), "float32")
    )

    builder.finalize()
    assert tguf_path.exists()
    assert tguf_path.stat().st_size > 0


def test_builder_finalize_creates_file(schema, tguf_path):
    if tguf_path.exists():
        tguf_path.unlink()

    tguf.TGUFBuilder(schema).finalize()
    assert tguf_path.exists()
