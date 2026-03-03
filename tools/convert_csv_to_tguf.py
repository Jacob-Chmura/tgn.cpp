import argparse
import struct
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Convert TG CSV data to TGUF binary format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog="""

EXPECTED CSV STRUCTURE:

1. Edges CSV (--edges):
   - Mandatory columns: 'src', 'dst', 'time'
   - Optional messages: 'msg_0', 'msg_1', ..., 'msg_n' (detected by 'msg_' prefix, must be uniform)
   - Optional negatives: 'neg_0', 'neg_1', ..., 'neg_k' (detected by 'neg_' prefix, must be uniform)

2. Node Labels CSV (--labels):
   - Mandatory columns: 'node_id', 'time'
   - Labels: 'node_y0', 'node_y1', ..., 'node_ym' (detected by 'node_y' prefix, must be uniform)

Note: The script uses 'wc -l' for fast row counting. Ensure the CSV ends with a newline.
""",
)
parser.add_argument("--edges", type=Path, required=True, help="Path to edges.csv")
parser.add_argument("--labels", type=Path, help="Path to optional node_labels.csv")
parser.add_argument("--output", type=Path, required=True, help="Output .tguf path")
parser.add_argument(
    "--batch_size", type=int, default=16384, help="Streaming batch size"
)

_TGUF_BIN = "./build/tools/tguf_cli/tguf_cli"


def main() -> None:
    args = parser.parse_args()

    if not args.edges.is_file():
        raise ValueError(f"Edges file {args.edges} not found or not a file")
    if args.labels is not None and not args.labels.is_file():
        raise ValueError(f"Labels file {args.labels} not found or not a file")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Verified inputs. Output will be written to: {args.output.resolve()}")

    n_edges, m_dim, n_neg, _ = get_csv_info(args.edges)
    print(f"Num edges: {n_edges}, Msg dim: {m_dim}, Num negatives: {n_neg}")

    n_labels, l_dim = 0, 0
    if args.labels:
        n_labels, _, _, l_dim = get_csv_info(args.labels)
        print(f"Num labels: {n_labels}, Label dim: {l_dim}")

    streamer = TGUFStreamer(args.output, n_edges, m_dim, n_neg, n_labels, l_dim)

    try:
        edge_chunks = (n_edges + args.batch_size - 1) // args.batch_size
        with tqdm(total=edge_chunks, desc="Edges", unit="batch") as pbar:
            pbar.set_postfix({"batch_size": args.batch_size})
            for chunk in pd.read_csv(args.edges, chunksize=args.batch_size):
                streamer.stream_edges(chunk)
                pbar.update(1)

        if n_labels > 0:
            label_chunks = (n_labels + args.batch_size - 1) // args.batch_size
            with tqdm(total=label_chunks, desc="Labels", unit="chunk") as pbar:
                pbar.set_postfix({"batch_size": args.batch_size})
                for chunk in pd.read_csv(args.labels, chunksize=args.batch_size):
                    streamer.stream_labels(chunk)
                    pbar.update(1)
        streamer.finalize()
        print(f"Successfully created {args.output.resolve()}")
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
    ) -> None:
        self.proc = subprocess.Popen([_TGUF_BIN, str(out_path)], stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open pipe to tguf_cli")

        # Send header
        header = struct.pack(
            "7Q", n_edges, m_dim, n_neg, n_labels, l_dim, val_start, test_start
        )
        self.proc.stdin.write(header)

        self.m_dim = m_dim
        self.n_neg = n_neg
        self.l_dim = l_dim

    def stream_edges(self, df: pd.DataFrame) -> None:
        batch_size = len(df)
        self.proc.stdin.write(b"E")
        self.proc.stdin.write(struct.pack("Q", batch_size))

        self._write_array(df["src"].values, np.int64)
        self._write_array(df["dst"].values, np.int64)
        self._write_array(df["time"].values, np.int64)

        msg_cols = [f"msg_{i}" for i in range(self.m_dim)]
        self._write_array(df[msg_cols].values, np.float32)

        if self.n_neg > 0:
            neg_cols = [f"neg_{i}" for i in range(self.n_neg)]
            self._write_array(df[neg_cols].values, np.int64)

    def stream_labels(self, df: pd.DataFrame) -> None:
        batch_size = len(df)
        self.proc.stdin.write(b"L")
        self.proc.stdin.write(struct.pack("Q", batch_size))

        self._write_array(df["node_id"].values, np.int64)
        self._write_array(df["time"].values, np.int64)

        label_cols = [f"node_y{i}" for i in range(self.l_dim)]
        self._write_array(df[label_cols].values, np.float32)

    def finalize(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"CLI failed with exit code {ret}")

    def _write_array(self, x: np.ndarray, dtype) -> None:  # type: ignore
        if x.dtype != dtype:
            x = x.astype(dtype)
        x = np.ascontiguousarray(x)
        self.proc.stdin.write(x.data)


def get_csv_info(path: Path) -> Tuple[int, ...]:
    print(f"Infering metadata from {path}...")
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
