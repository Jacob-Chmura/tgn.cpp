import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np
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

_TGUF_BIN = "./build/tools/tguf_cli/tguf_cli"


def main() -> None:
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = download_dataset(args.name)
    data = ds.full_data

    # Edge data
    src, dst, ts = data["sources"], data["destinations"], data["timestamps"]
    edge_feat = data["edge_feat"]
    m_dim = edge_feat.shape[1] if edge_feat is not None else 0
    n_edges = len(src)

    # Pre-computed negatives (for link prediction)
    n_neg, full_negs = 0, None
    if args.name.startswith("tgbl-"):
        ns = NegativeEdgeSampler(dataset_name=args.name)
        v = (
            f"_v{DATA_VERSION_DICT[args.name]}"
            if DATA_VERSION_DICT.get(args.name, 1) > 1
            else ""
        )

        # Max number of negative we parse is 1000 per positive link
        full_negs = np.full((len(src), 1000), -1, dtype=np.int32)

        for split in ["val", "test"]:
            print(f"Processing negatives for {split}...")
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

    val_start = int(np.argmax(ds.val_mask))
    test_start = int(np.argmax(ds.test_mask))
    streamer = TGUFStreamer(
        args.output, n_edges, m_dim, n_neg, n_labels, l_dim, val_start, test_start
    )

    try:
        edge_chunks = (n_edges + args.batch_size - 1) // args.batch_size
        with tqdm(total=edge_chunks, desc="Edges", unit="chunk") as pbar:
            pbar.set_postfix({"bsize": args.batch_size})
            for i in range(0, len(src), args.batch_size):
                end = i + args.batch_size

                streamer.stream_edge_batch(
                    src[i:end],
                    dst[i:end],
                    ts[i:end],
                    edge_feat[i:end] if m_dim > 0 else None,
                    full_negs[i:end] if n_neg > 0 else None,
                )
                pbar.update(1)

        if n_labels > 0:
            label_chunks = (n_labels + args.batch_size - 1) // args.batch_size
            with tqdm(total=label_chunks, desc="Labels", unit="chunk") as pbar:
                pbar.set_postfix({"bsize": args.batch_size})
                for i in range(0, n_labels, args.batch_size):
                    end = i + args.batch_size
                    batch = label_data[i:end]

                    # Binary Order: node_id, t, y_true
                    # (Note: batch[:, 0] is 'time', batch[:, 1] is 'node_id' based on previous logic)
                    streamer.stream_label_batch(
                        ts=batch[:, 0], nodes=batch[:, 1], labels=batch[:, 2:]
                    )
                    pbar.update(1)

        streamer.finalize()
        print(f"Successfully created: {args.output.resolve()}")
    except Exception as e:
        print(f"Error during streaming: {e}")
        streamer.proc.terminate()


class TGUFStreamer:
    def __init__(
        self,
        out_path: Path,
        n_edges: int,
        m_dim: int,
        n_neg: int,
        n_labels: int,
        l_dim: int,
        val_start: int,
        test_start: int,
    ) -> None:
        cmd = [_TGUF_BIN]
        cmd += ["--out", str(out_path)]
        cmd += ["--n_edges", str(n_edges)]
        cmd += ["--m_dim", str(m_dim)]
        cmd += ["--n_neg", str(n_neg)]
        cmd += ["--n_labels", str(n_labels)]
        cmd += ["--l_dim", str(l_dim)]
        cmd += ["--val_start", str(val_start)]
        cmd += ["--test_start", str(test_start)]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError(f"Failed to open stdin for TGUF CLI process {_TGUF_BIN}")

        self.m_dim = m_dim
        self.n_neg = n_neg
        self.l_dim = l_dim

    def stream_edge_batch(self, src, dst, ts, msg, negs):
        batch_size = len(src)
        self.proc.stdin.write(b"E")
        self.proc.stdin.write(struct.pack("Q", batch_size))

        self.write_col(src, np.int64)
        self.write_col(dst, np.int64)
        self.write_col(ts, np.int64)

        if self.m_dim > 0:
            self.write_col(msg, np.float32)
        if self.n_neg > 0:
            self.write_col(negs, np.int64)

    def stream_label_batch(self, ts, nodes, labels):
        batch_size = len(ts)
        self.proc.stdin.write(b"L")
        self.proc.stdin.write(struct.pack("Q", batch_size))
        self.write_col(nodes, np.int64)
        self.write_col(ts, np.int64)
        self.write_col(labels, np.float32)

    def finalize(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        retcode = self.proc.wait()
        if retcode != 0:
            raise RuntimeError(f"CLI failed with exit code {retcode}")

    def write_col(self, x: np.ndarray, dtype) -> None:  # type: ignore
        self.proc.stdin.write(x.astype(dtype).tobytes())


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
