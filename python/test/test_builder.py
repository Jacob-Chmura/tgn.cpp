import numpy as np
import pytest
import tguf
import torch


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_builder_append_edges(schema, lib, tguf_path, get_data):
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
def test_builder_append_labels(schema, lib, tguf_path, get_data):
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
def test_builder_append_node_feats(schema, lib, tguf_path, get_data):
    builder = tguf.TGUFBuilder(schema)

    builder.append_node_feats(
        get_data(lib, [0, 1], "int64"), get_data(lib, np.random.randn(2, 4), "float32")
    )

    builder.finalize()
    assert tguf_path.exists()
    assert tguf_path.stat().st_size > 0


def test_builder_to_store_roundtrip(schema):
    num_edges = 10
    src = np.arange(num_edges, dtype=np.int64)
    msgs = np.random.randn(num_edges, schema.msg_dim).astype(np.float32)

    builder = tguf.TGUFBuilder(schema)
    batch = tguf.Batch(src=src, dst=src + 100, time=src * 10, msg=msgs)
    builder.append_edges(batch)

    n_ids = np.array([0, 1, 2], dtype=np.int64)
    n_feats = np.random.randn(3, schema.node_feat_dim).astype(np.float32)
    builder.append_node_feats(n_ids, n_feats)
    builder.finalize()

    store = tguf.TGStore.from_tguf(schema.path)
    assert store.edge_count == num_edges

    b_out = store.get_batch(start=0, size=num_edges)
    torch.testing.assert_close(b_out.src, torch.from_numpy(src))
    torch.testing.assert_close(b_out.dst, torch.from_numpy(src + 100))
    torch.testing.assert_close(b_out.time, torch.from_numpy(src * 10))
    torch.testing.assert_close(b_out.msg, torch.from_numpy(msgs))
    assert b_out.neg_dst is None

    nf_out = store.gather_node_feats(n_ids)
    torch.testing.assert_close(nf_out, torch.from_numpy(n_feats))
