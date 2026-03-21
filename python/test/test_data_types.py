import numpy as np
import pytest
import tguf
import torch


def test_index_range():
    ir = tguf.IndexRange(10, 50)
    assert ir.start == 10
    assert ir.end == 50
    assert ir.size == 40


def test_neg_strategy_enum():
    assert hasattr(tguf.NegStrategy, "None_")
    assert hasattr(tguf.NegStrategy, "Random")
    assert hasattr(tguf.NegStrategy, "PreComputed")


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_batch_init(lib, get_data):
    tguf.Batch(
        src=get_data(lib, [0, 1, 2], "int64"),
        dst=get_data(lib, [3, 4, 5], "int64"),
        time=get_data(lib, [10, 11, 12], "int64"),
        msg=get_data(lib, np.random.randn(3, 8), "float32"),
    )


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_batch_init_with_negs(lib, get_data):
    tguf.Batch(
        src=get_data(lib, [0, 1, 2], "int64"),
        dst=get_data(lib, [3, 4, 5], "int64"),
        time=get_data(lib, [10, 11, 12], "int64"),
        msg=get_data(lib, np.random.randn(3, 8), "float32"),
        neg_dst=get_data(lib, [3, 4, 5], "int64"),
    )


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_batch_get(lib, get_data):
    raw_src = [0, 1, 2]
    raw_dst = [3, 4, 5]
    raw_time = [10, 11, 12]
    raw_msg = np.random.randn(3, 8).astype(np.float32)
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
        assert torch.equal(tensor, torch.tensor(expected, dtype=torch.int64))

    assert batch.neg_dst is not None
    assert torch.is_tensor(batch.neg_dst)
    assert torch.equal(batch.neg_dst, torch.tensor(raw_negs, dtype=torch.int64))


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_label_event_init(lib, get_data):
    tguf.LabelEvent(
        n_id=get_data(lib, [0, 1, 2], "int64"),
        target=get_data(lib, np.random.randn(3, 8), "float32"),
    )


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_label_event_get(lib, get_data):
    raw_n_ids = [10, 20, 30]
    raw_targets = np.array([[1.0], [5.0], [10.0]], dtype=np.float32)

    le = tguf.LabelEvent(
        n_id=get_data(lib, raw_n_ids, "int64"),
        target=get_data(lib, raw_targets, "float32"),
    )

    assert torch.is_tensor(le.n_id)
    assert torch.equal(le.n_id, torch.tensor(raw_n_ids, dtype=torch.int64))

    assert torch.is_tensor(le.target)
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
    updates = {
        "path": "new_path.tguf",
        "edge_capacity": 1000,
        "label_capacity": 50,
        "msg_dim": 128,
        "val_start": 800,
    }

    for key, value in updates.items():
        setattr(schema, key, value)
        assert getattr(schema, key) == value
