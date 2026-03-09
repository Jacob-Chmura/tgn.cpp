import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple

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

Note: msg_dim, num_negatives and node_y dimensions must be uniform across the data.
Note: The script uses 'wc -l' for fast row counting.
""",
)
parser.add_argument("--edges", type=Path, required=True, help="Path to edges.csv")
parser.add_argument("--labels", type=Path, help="Path to optional node_labels.csv")
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

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_edges, m_dim, n_neg, _ = get_csv_info(args.edges)
    n_labels, l_dim = 0, 0
    if args.labels:
        n_labels, _, _, l_dim = get_csv_info(args.labels)
    neg_capacity = n_edges if n_neg > 0 else 0

    streamer = TGUFStreamer(
        args.output, n_edges, m_dim, n_neg, n_labels, l_dim, neg_capacity
    )

    msg_cols = [f"msg_{i}" for i in range(m_dim)]
    neg_cols = [f"neg_{i}" for i in range(n_neg)]
    label_cols = [f"node_y{i}" for i in range(l_dim)]

    try:
        edge_chunks = (n_edges + args.batch_size - 1) // args.batch_size
        with tqdm(total=edge_chunks, desc="Appending Edges", unit="batch") as pbar:
            pbar.set_postfix({"batch_size": args.batch_size})
            for chunk in pd.read_csv(args.edges, chunksize=args.batch_size):
                streamer.stream_edge_batch(
                    src=chunk["src"].values,
                    dst=chunk["dst"].values,
                    ts=chunk["time"].values,
                    msg=chunk[msg_cols].values,
                    negs=chunk[neg_cols].values if n_neg > 0 else None,
                )
                pbar.update(1)

        if n_labels > 0:
            label_chunks = (n_labels + args.batch_size - 1) // args.batch_size
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

        streamer.finalize()
    except Exception as e:
        print(f"Error during streaming: {e}")
        streamer.proc.terminate()


def get_csv_info(path: Path) -> Tuple[int, ...]:
    preview = pd.read_csv(path, nrows=1, comment="#")
    cols = preview.columns.tolist()

    m_dim = sum(1 for c in cols if c.startswith("msg_"))
    n_neg = sum(1 for c in cols if c.startswith("neg_"))
    l_dim = sum(1 for c in cols if c.startswith("node_y"))

    # Count rows using a system call (subtract 1 for the header)
    n_rows = int(subprocess.check_output(["wc", "-l", path]).split()[0]) - 1

    return n_rows, m_dim, n_neg, l_dim


if __name__ == "__main__":
    main()
