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
        # val_start=8,
        # test_start=9,
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
def test_batch_get(lib):
    num_samples = 3
    msg_dim = 8
    num_negs = 2

    raw_src = [0, 1, 2]
    raw_dst = [3, 4, 5]
    raw_time = [10, 11, 12]
    raw_msg = np.random.randn(num_samples, msg_dim).astype(np.float32)
    raw_negs = np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int64)

    batch = tguf.Batch(
        src=get_data(lib, raw_src, "int64"),
        dst=get_data(lib, raw_dst, "int64"),
        time=get_data(lib, raw_time, "int64"),
        msg=get_data(lib, raw_msg, "float32"),
        neg_dst=get_data(lib, raw_negs, "int64"),
    )

    for field, expected in [("src", raw_src), ("dst", raw_dst), ("time", raw_time)]:
        tensor = getattr(batch, field)
        assert torch.is_tensor(tensor)
        assert tensor.dtype == torch.int64
        assert torch.equal(tensor, torch.tensor(expected, dtype=torch.int64))

    assert batch.neg_dst is not None
    assert torch.is_tensor(batch.neg_dst)
    assert batch.neg_dst.dtype == torch.int64
    assert batch.neg_dst.ndim == 2
    assert batch.neg_dst.shape == (num_samples, num_negs)
    assert torch.equal(batch.neg_dst, torch.tensor(raw_negs, dtype=torch.int64))


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_label_event_init(lib):
    tguf.LabelEvent(
        n_id=get_data(lib, [0, 1, 2], "int64"),
        target=get_data(lib, np.random.randn(3, 8), "float32"),
    )


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_label_event_get(lib):
    num_nodes = 3
    label_dim = 1
    raw_n_ids = [10, 20, 30]
    raw_targets = np.array([[1.0], [5.0], [10.0]], dtype=np.float32)

    le = tguf.LabelEvent(
        n_id=get_data(lib, raw_n_ids, "int64"),
        target=get_data(lib, raw_targets, "float32"),
    )

    assert torch.is_tensor(le.n_id)
    assert le.n_id.dtype == torch.int64
    assert le.n_id.ndim == 1
    assert le.n_id.shape == (num_nodes,)
    assert torch.equal(le.n_id, torch.tensor(raw_n_ids, dtype=torch.int64))

    assert torch.is_tensor(le.target)
    assert le.target.dtype == torch.float32
    assert le.target.ndim == 2
    assert le.target.shape == (num_nodes, label_dim)
    assert torch.allclose(le.target, torch.from_numpy(raw_targets), atol=1e-6)


def test_schema_init():
    schema = tguf.TGUFSchema(path="test.tguf", edge_capacity=500, msg_dim=16)

    assert schema.path == "test.tguf"
    assert schema.edge_capacity == 500
    assert schema.msg_dim == 16
    assert schema.label_dim == 0
    assert schema.node_feat_capacity == 0
    assert schema.node_feat_dim == 0
    assert schema.negatives_start_e_id == 0
    assert schema.negatives_per_edge == 0
    assert schema.val_start is None
    assert schema.test_start is None


def test_schema_get_set():
    schema = tguf.TGUFSchema(path="test.tguf", edge_capacity=500, msg_dim=16)

    schema.path = "new_path.tguf"
    assert schema.path == "new_path.tguf"

    schema.edge_capacity = 1000
    assert schema.edge_capacity == 1000

    schema.label_capacity = 50
    assert schema.label_capacity == 50

    schema.node_feat_capacity = 250
    assert schema.node_feat_capacity == 250

    schema.msg_dim = 128
    assert schema.msg_dim == 128

    schema.label_dim = 32
    assert schema.label_dim == 32

    schema.node_feat_dim = 64
    assert schema.node_feat_dim == 64

    schema.negatives_start_e_id = 5000
    assert schema.negatives_start_e_id == 5000

    schema.negatives_per_edge = 10
    assert schema.negatives_per_edge == 10

    schema.val_start = 800
    assert schema.val_start == 800

    schema.test_start = 900
    assert schema.test_start == 900


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


def test_tgstore_split_logic(schema):
    schema.val_start = 8
    schema.test_start = 9
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(10, dtype=np.int64),
        dst=np.zeros(10, dtype=np.int64),
        time=np.arange(10, dtype=np.int64),
        msg=np.zeros((10, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    # Verify split boundaries
    assert store.train_split.start == 0
    assert store.train_split.end == 8
    assert store.val_split.start == 8
    assert store.val_split.size == 1  # 9 - 8
    assert store.test_split.start == 9
    assert store.test_split.size == 1  # 10 - 9


def test_tgstore_vectorized_gathers(schema):
    times = np.array([100, 200, 300, 400], dtype=np.int64)
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(4, dtype=np.int64),
        dst=np.zeros(4, dtype=np.int64),
        time=times,
        msg=np.zeros((4, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    indices = torch.tensor([3, 0, 1], dtype=torch.long)
    t_out = store.gather_timestamps(indices)

    assert torch.equal(t_out, torch.tensor([400, 100, 200], dtype=torch.long))


def test_tgstore_label_event_and_cutoff(schema):
    builder = tguf.TGUFBuilder(schema)
    # Add 10 edges at times 0, 10, 20...
    builder.append_edges(
        tguf.Batch(
            src=np.zeros(10),
            dst=np.zeros(10),
            time=np.arange(10) * 10,
            msg=np.zeros((10, schema.msg_dim)),
        )
    )
    # Add a label at time 45 (should cutoff at edge_id 4, which is time 40)
    builder.append_labels(
        n_id=np.array([1]), time=np.array([45]), target=np.array([[1]])
    )
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)

    # Verify cutoff
    cutoff = store.get_edge_cutoff_for_label_event(0)
    assert cutoff == 5  # 5 edges (0, 10, 20, 30, 40) are <= time 45

    # Verify label data retrieval
    label = store.get_label_event(0)
    assert label.n_id[0] == 1
    assert label.target.shape == (1, schema.label_dim)


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


def test_builder_to_store_roundtrip(schema, tguf_path):
    num_edges = 10
    src = np.arange(num_edges, dtype=np.int64)
    dst = src + 100
    times = src * 10
    msgs = np.random.randn(num_edges, schema.msg_dim).astype(np.float32)

    schema.path = str(tguf_path)

    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(src=src, dst=dst, time=times, msg=msgs)
    builder.append_edges(batch)

    n_ids = np.array([0, 1, 2], dtype=np.int64)
    n_feats = np.random.randn(3, schema.node_feat_dim).astype(np.float32)
    builder.append_node_feats(n_ids, n_feats)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)
    assert store.edge_count == num_edges

    b_out = store.get_batch(start=0, size=num_edges)
    assert np.allclose(b_out.msg.numpy(), msgs)
    assert np.array_equal(b_out.src.numpy(), src)

    nf_out = store.gather_node_feats(n_ids)
    assert np.allclose(nf_out.numpy(), n_feats, atol=1e-6)
