import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from _tguf_streamer import TGUFStreamer  # noqa: E402

parser = argparse.ArgumentParser(
    description="Convert CSV data to TGUF binary format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog="""

EXPECTED CSV STRUCTURE

1. Edges CSV (--edges):
   - Mandatory columns: 'src', 'dst', 'time'
   - Optional messages: 'msg_0', 'msg_1', ..., 'msg_n' (detected by 'msg_' prefix)
   - Optional negatives: 'neg_0', 'neg_1', ..., 'neg_k' (detected by 'neg_' prefix)

2. Node Labels CSV (--labels):
   - Mandatory columns: 'node_id', 'time'
   - Labels: 'node_y0', 'node_y1', ..., 'node_ym' (detected by 'node_y' prefix)

2. Node Feats CSV (--node-feats):
   - Mandatory columns: 'node_id'
   - Feats: 'node_x0', 'node_x1', ..., 'node_xm' (detected by 'node_x' prefix)

Note: msg_dim, num_negatives, node_x and node_y dimensions must be uniform across the data.
Note: The script uses 'wc -l' for fast row counting.
""",
)
parser.add_argument("--edges", type=Path, required=True, help="Path to edges.csv")
parser.add_argument("--labels", type=Path, help="Path to optional node_labels.csv")
parser.add_argument("--node-feats", type=Path, help="Path to optional node_feats.csv")
parser.add_argument("--output", type=Path, required=True, help="Output .tguf path")
parser.add_argument(
    "--batch_size", type=int, default=16384, help="Streaming batch size"
)


def main() -> None:
    args = parser.parse_args()

    if not args.edges.is_file():
        raise ValueError(f"Edges file {args.edges} not found or not a file")
    if args.labels is not None and not args.labels.is_file():
        raise ValueError(f"Labels file {args.labels} not found or not a file")
    if args.node_feats is not None and not args.node_feats.is_file():
        raise ValueError(f"Node Feats file {args.node_feats} not found or not a file")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    e_info = get_csv_info(args.edges)
    global_max_id = e_info["max_id"]

    l_info = {"n_rows": 0, "label_dim": 0}
    if args.labels:
        l_info = get_csv_info(args.labels)
        global_max_id = max(global_max_id, l_info["max_id"])

    n_info = {"n_rows": 0, "feat_dim": 0}
    if args.node_feats:
        n_info = get_csv_info(args.node_feats)
        global_max_id = max(global_max_id, n_info["max_id"])

    streamer = TGUFStreamer(
        args.output,
        n_edges=e_info["n_rows"],
        m_dim=e_info["msg_dim"],
        n_neg=e_info["n_neg"],
        n_labels=l_info["n_rows"],
        l_dim=l_info["label_dim"],
        n_nodes=global_max_id + 1,
        n_dim=n_info["feat_dim"],
    )

    msg_cols = [f"msg_{i}" for i in range(e_info["msg_dim"])]
    neg_cols = [f"neg_{i}" for i in range(e_info["n_neg"])]
    label_cols = [f"node_y{i}" for i in range(l_info["label_dim"])]
    node_feat_cols = [f"node_x{i}" for i in range(n_info["feat_dim"])]

    try:
        edge_chunks = (e_info["n_rows"] + args.batch_size - 1) // args.batch_size
        with tqdm(total=edge_chunks, desc="Appending Edges", unit="batch") as pbar:
            pbar.set_postfix({"batch_size": args.batch_size})
            for chunk in pd.read_csv(args.edges, chunksize=args.batch_size):
                streamer.stream_edge_batch(
                    src=chunk["src"].values,
                    dst=chunk["dst"].values,
                    ts=chunk["time"].values,
                    msg=chunk[msg_cols].values,
                    negs=chunk[neg_cols].values if e_info["n_neg"] > 0 else None,
                )
                pbar.update(1)

        if l_info["n_rows"] > 0:
            label_chunks = (l_info["n_rows"] + args.batch_size - 1) // args.batch_size
            with tqdm(
                total=label_chunks, desc="Appending Labels", unit="chunk"
            ) as pbar:
                pbar.set_postfix({"batch_size": args.batch_size})
                for chunk in pd.read_csv(args.labels, chunksize=args.batch_size):
                    streamer.stream_label_batch(
                        nodes=chunk["node_id"].values,
                        ts=chunk["time"].values,
                        labels=chunk[label_cols].values,
                    )
                    pbar.update(1)

        if n_info["n_rows"] > 0:
            node_feat_chunks = (
                n_info["n_rows"] + args.batch_size - 1
            ) // args.batch_size
            with tqdm(
                total=node_feat_chunks, desc="Appending Node Feats", unit="chunk"
            ) as pbar:
                pbar.set_postfix({"batch_size": args.batch_size})
                for chunk in pd.read_csv(args.node_feats, chunksize=args.batch_size):
                    streamer.stream_node_feat_batch(
                        nodes=chunk["node_id"].values,
                        feats=chunk[node_feat_cols].values,
                    )
                    pbar.update(1)

        streamer.finalize()
    except Exception as e:
        print(f"Error during streaming: {e}")
        streamer.proc.terminate()


def get_csv_info(path: Path) -> Dict[str, int]:
    preview = pd.read_csv(path, nrows=1, comment="#")
    cols = preview.columns.tolist()
    info = {
        "n_rows": int(subprocess.check_output(["wc", "-l", path]).split()[0]) - 1,
        "msg_dim": sum(1 for c in cols if c.startswith("msg_")),
        "n_neg": sum(1 for c in cols if c.startswith("neg_")),
        "label_dim": sum(1 for c in cols if c.startswith("node_y")),
        "feat_dim": sum(1 for c in cols if c.startswith("node_x")),
        "max_id": 0,
    }

    def get_max_id_from_csv(
        path: Path, id_cols: List[str], batch_size: int = 1_000_000
    ) -> int:
        max_val = 0
        for chunk in pd.read_csv(path, usecols=id_cols, chunksize=batch_size):
            max_val = max(max_val, int(chunk.max().max()))
        return max_val

    # Find the maximum ID to determine capacity
    id_cols = [c for c in ["src", "dst", "node_id"] if c in cols]
    if id_cols:
        info["max_id"] = get_max_id_from_csv(path, id_cols)
    return info


if __name__ == "__main__":
    main()
