import csv
from pathlib import Path

import numpy as np
import pytest
import torch

import tguf


@pytest.fixture
def resource_dir():
    return Path(__file__).parent / "resources" / "csv_tguf_roundtrip"


@pytest.fixture
def output_tguf(tmp_path):
    return tmp_path / "out.tguf"


def load_csv_data(path: Path):
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def test_csv_tguf_roundtrip(resource_dir, output_tguf):
    e_headers, e_rows = load_csv_data(resource_dir / "edges.csv")
    l_headers, l_rows = load_csv_data(resource_dir / "labels.csv")
    n_headers, n_rows = load_csv_data(resource_dir / "node_feats.csv")

    msg_cols = [c for c in e_headers if c.startswith("msg_")]
    neg_cols = [c for c in e_headers if c.startswith("neg_")]
    label_cols = [c for c in l_headers if c.startswith("node_y")]
    feat_cols = [c for c in n_headers if c.startswith("node_x")]

    all_ids = []
    for r in e_rows:
        all_ids.extend([int(r["src"]), int(r["dst"])])
    for r in l_rows:
        all_ids.append(int(r["node_id"]))
    for r in n_rows:
        all_ids.append(int(r["node_id"]))
    max_id = max(all_ids)

    schema = tguf.TGUFSchema(
        path=str(output_tguf),
        edge_capacity=len(e_rows),
        msg_dim=len(msg_cols),
        label_dim=len(label_cols),
        node_feat_capacity=int(max_id + 1),
        node_feat_dim=len(feat_cols),
        label_capacity=len(l_rows),
        negatives_per_edge=len(neg_cols),
    )
    builder = tguf.TGUFBuilder(schema)

    builder.append_edges(
        tguf.Batch(
            src=np.array([int(r["src"]) for r in e_rows], dtype=np.int64),
            dst=np.array([int(r["dst"]) for r in e_rows], dtype=np.int64),
            time=np.array([int(r["time"]) for r in e_rows], dtype=np.int64),
            msg=np.array(
                [[float(r[c]) for c in msg_cols] for r in e_rows], dtype=np.float32
            ),
            neg_dst=np.array(
                [[int(r[c]) for c in neg_cols] for r in e_rows], dtype=np.int64
            )
            if neg_cols
            else None,
        )
    )

    builder.append_labels(
        n_id=np.array([int(r["node_id"]) for r in l_rows], dtype=np.int64),
        time=np.array([int(r["time"]) for r in l_rows], dtype=np.int64),
        target=np.array(
            [[float(r[c]) for c in label_cols] for r in l_rows], dtype=np.float32
        ),
    )

    builder.append_node_feats(
        n_id=np.array([int(r["node_id"]) for r in n_rows], dtype=np.int64),
        node_feat=np.array(
            [[float(r[c]) for c in feat_cols] for r in n_rows], dtype=np.float32
        ),
    )

    builder.finalize()

    store = tguf.TGStore.from_tguf(str(output_tguf))

    assert store.edge_count == 3
    assert store.node_count == 31
    assert store.msg_dim == 2
    assert store.label_dim == 2
    assert store.node_feat_dim == 3

    batch = store.get_batch(0, 3, strategy=tguf.NegStrategy.PreComputed)
    torch.testing.assert_close(batch.src, torch.tensor([1, 2, 3], dtype=torch.int64))
    torch.testing.assert_close(batch.dst, torch.tensor([20, 30, 10], dtype=torch.int64))
    torch.testing.assert_close(batch.time, torch.tensor([5, 10, 15], dtype=torch.int64))

    expected_negs = torch.tensor([[9, 8], [7, 6], [5, 4]], dtype=torch.int64)
    torch.testing.assert_close(batch.neg_dst, expected_negs)

    label0 = store.get_label_event(0)
    assert label0.n_id[0] == 1
    torch.testing.assert_close(
        label0.target[0], torch.tensor([1.0, 0.0], dtype=torch.float32)
    )

    assert store.get_edge_cutoff_for_label_event(0) == 2
    assert store.get_edge_cutoff_for_label_event(1) == 3

    n_ids = torch.arange(store.node_count, dtype=torch.int64)
    node_feats = store.gather_node_feats(n_ids)

    for i in range(len(n_ids)):
        if i % 5 == 0 and i <= 20:
            mult = i / 5.0
            expected = torch.tensor(
                [1.0 * mult, 2.0 * mult, 3.0 * mult], dtype=torch.float32
            )
            torch.testing.assert_close(node_feats[i], expected)
        else:
            torch.testing.assert_close(node_feats[i], torch.zeros(3))
