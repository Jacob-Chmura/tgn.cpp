import tguf
import numpy as np
import torch


def test_tgstore_from_memory(get_data):
    num_edges = 20
    batch = tguf.Batch(
        src=get_data("numpy", np.arange(num_edges), "int64"),
        dst=get_data("numpy", np.arange(num_edges) + 1, "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.random.randn(num_edges, 8), "float32"),
    )
    store = tguf.TGStore.from_memory(edges=batch, val_start=15, test_start=18)

    assert store.edge_count == num_edges
    assert store.node_count == num_edges + 1
    assert store.msg_dim == 8
    assert store.label_dim == 0
    assert store.train_split.end == 15
    assert store.val_split.size == 3
    assert store.test_split.start == 18

    b = store.get_batch(0, 5, tguf.NegStrategy.None_)
    assert torch.is_tensor(b.src)
    assert torch.is_tensor(b.dst)
    assert torch.is_tensor(b.time)
    assert torch.is_tensor(b.msg)
    assert b.neg_dst is None


def test_tgstore_from_tguf(schema, get_data):
    builder = tguf.TGUFBuilder(schema)
    num_edges = 20
    batch = tguf.Batch(
        src=get_data("numpy", np.arange(num_edges), "int64"),
        dst=get_data("numpy", np.arange(num_edges) + 1, "int64"),
        time=get_data("numpy", np.arange(num_edges) * 10, "int64"),
        msg=get_data("numpy", np.random.randn(num_edges, 8), "float32"),
    )
    builder.append_edges(batch)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path, val_start=15, test_start=18)

    assert store.edge_count == num_edges
    assert store.node_count == num_edges + 1
    assert store.msg_dim == 8
    assert store.label_dim == 0
    assert store.train_split.end == 15
    assert store.val_split.size == 3
    assert store.test_split.start == 18

    b = store.get_batch(0, 5, tguf.NegStrategy.None_)
    assert torch.is_tensor(b.src)
    assert torch.is_tensor(b.dst)
    assert torch.is_tensor(b.time)
    assert torch.is_tensor(b.msg)
    assert b.neg_dst is None


def test_tgstore_node_features(get_data):
    num_nodes = 5
    feat_dim = 4
    feats = np.random.randn(num_nodes, feat_dim).astype(np.float32)

    batch = tguf.Batch(
        src=get_data("numpy", [0], "int64"),
        dst=get_data("numpy", [1], "int64"),
        time=get_data("numpy", [0], "int64"),
        msg=get_data("numpy", np.zeros((1, 8)), "float32"),
    )

    store = tguf.TGStore.from_memory(
        edges=batch,
        node_feats=torch.from_numpy(feats),
        label_n_id=None,
        label_time=None,
        label_target=None,
        val_start=None,
        test_start=None,
    )

    query_nodes = torch.tensor([4, 0, 2], dtype=torch.long)
    out = store.gather_node_feats(query_nodes)

    torch.testing.assert_close(out, torch.from_numpy(feats[[4, 0, 2]]))


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

    cutoff = store.get_edge_cutoff_for_label_event(0)
    assert cutoff == 5  # 5 edges (0, 10, 20, 30, 40) are <= time 45

    label = store.get_label_event(0)
    assert label.n_id[0] == 1
    assert label.target.shape == (1, schema.label_dim)


def test_tensor_independence(schema):
    num_edges = 10
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(num_edges, dtype=np.int64),
        dst=np.zeros(num_edges, dtype=np.int64),
        time=np.arange(num_edges, dtype=np.int64) * 10,
        msg=np.zeros((num_edges, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)

    indices = torch.tensor([0, 1, 2], dtype=torch.long)
    t_out = store.gather_timestamps(indices)

    # Delete the store (triggering C++ destructors)
    del store

    # Try to access the data. If the capsule worked, this remains valid.
    assert t_out[0] == 0
    assert t_out.sum() == 30  # (0 + 10 + 20)


def test_empty_gather(schema):
    num_edges = 10
    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(
        src=np.zeros(num_edges, dtype=np.int64),
        dst=np.zeros(num_edges, dtype=np.int64),
        time=np.arange(num_edges, dtype=np.int64) * 10,
        msg=np.zeros((num_edges, schema.msg_dim), dtype=np.float32),
    )
    builder.append_edges(batch)
    builder.finalize()
    store = tguf.TGStore.from_tguf(schema.path)
    empty_indices = torch.tensor([], dtype=torch.long)
    t_out = store.gather_timestamps(empty_indices)

    assert t_out.shape[0] == 0
    assert torch.is_tensor(t_out)
