import argparse
from pathlib import Path

import numpy as np
import tguf
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Download TGB dataset directly to TGUF",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--name", type=str, required=True, help="TGB dataset (e.g., tgbl-wiki)"
)
parser.add_argument("--output", type=Path, required=True, help="Output .tguf path")
parser.add_argument(
    "--batch_size", type=int, default=16384, help="Streaming batch size"
)


def main() -> None:
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = download_dataset(args.name)
    data = ds.full_data

    src, dst, ts = data["sources"], data["destinations"], data["timestamps"]
    edge_feat = data["edge_feat"]
    m_dim = edge_feat.shape[1] if edge_feat is not None else 0
    n_edges = len(src)

    # Pre-computed negatives (for link prediction)
    n_neg, full_negs = 0, None
    if args.name.startswith("tgbl-"):
        ns = NegativeEdgeSampler(dataset_name=args.name)
        v = ""
        if DATA_VERSION_DICT.get(args.name, 1) > 1:
            v = f"_v{DATA_VERSION_DICT[args.name]}"

        # Max number of negative we parse is 1000 per positive link
        full_negs = np.full((len(src), 1000), -1, dtype=np.int32)

        for split in ["val", "test"]:
            print(f"Processing negatives for {split} (TGB NegativeSampler is slow)...")
            mask = ds.val_mask if split == "val" else ds.test_mask
            ns_path = f"{PROJ_DIR}datasets/{args.name.replace('-', '_')}/{args.name}_{split}_ns{v}.pkl"
            ns.load_eval_set(fname=ns_path, split_mode=split)
            negs = ns.query_batch(src[mask], dst[mask], ts[mask], split_mode=split)

            # Update n_neg based on minimum available across batches
            cur_min = min(len(x) for x in negs)
            n_neg = cur_min if n_neg == 0 else min(n_neg, cur_min)
            full_negs[mask, :n_neg] = [x[:n_neg] for x in negs]

        full_negs = full_negs[:, :n_neg]

    # Node labels (for node prediction)
    n_labels, l_dim, label_data = 0, 0, None
    if args.name.startswith("tgbn-"):
        rows = []
        for t, node_dict in data["node_label_dict"].items():
            for node_id, y_true in node_dict.items():
                rows.append([t, node_id, *y_true])

        label_data = np.array(rows)
        n_labels = len(label_data)
        l_dim = label_data.shape[1] - 2  # Peek to get dimension: t, id, [labels...]

    # Bake in pre-defined splits into TGUF header
    val_start = int(np.argmax(ds.val_mask))
    test_start = int(np.argmax(ds.test_mask))

    neg_start_e_id = val_start if full_negs is not None else 0

    schema = tguf.TGUFSchema(
        path=str(args.output),
        edge_capacity=n_edges,
        msg_dim=m_dim,
        label_dim=l_dim,
        label_capacity=n_labels,
        negatives_per_edge=n_neg,
        negatives_start_e_id=neg_start_e_id,
        val_start=val_start,
        test_start=test_start,
    )
    builder = tguf.TGUFBuilder(schema)

    try:
        edge_chunks = (n_edges + args.batch_size - 1) // args.batch_size
        with tqdm(total=edge_chunks, desc="Appending Edges", unit="batch") as pbar:
            pbar.set_postfix({"batch_size": args.batch_size})
            for i in range(0, len(src), args.batch_size):
                end = i + args.batch_size
                batch = tguf.Batch(
                    src=np.ascontiguousarray(src[i:end], dtype="int64"),
                    dst=np.ascontiguousarray(dst[i:end], dtype="int64"),
                    time=np.ascontiguousarray(ts[i:end], dtype="int64"),
                    msg=np.ascontiguousarray(edge_feat[i:end], dtype="float32"),
                    neg_dst=np.ascontiguousarray(full_negs[i:end], dtype="int64")
                    if full_negs is not None
                    else None,
                )
                builder.append_edges(batch)
                pbar.update(1)

        if n_labels > 0:
            label_chunks = (n_labels + args.batch_size - 1) // args.batch_size
            assert label_data is not None
            with tqdm(
                total=label_chunks, desc="Appending Labels", unit="batch"
            ) as pbar:
                pbar.set_postfix({"batch_size": args.batch_size})
                for i in range(0, n_labels, args.batch_size):
                    end = i + args.batch_size
                    builder.append_labels(
                        n_id=np.ascontiguousarray(label_data[i:end, 0], dtype="int64"),
                        time=np.ascontiguousarray(label_data[i:end, 1], dtype="int64"),
                        target=np.ascontiguousarray(
                            label_data[i:end, 2:], dtype="float32"
                        ),
                    )
                    pbar.update(1)

        builder.finalize()
    except Exception as e:
        print(f"Error during streaming: {e}")


def download_dataset(name: str) -> LinkPropPredDataset | NodePropPredDataset:
    print(f"Downloading {name}...")
    if name.startswith("tgbl-"):
        return LinkPropPredDataset(name=name)
    elif name.startswith("tgbn-"):
        return NodePropPredDataset(name=name)
    else:
        raise ValueError(f"Unsupported tgb dataset: {name}")


if __name__ == "__main__":
    main()
