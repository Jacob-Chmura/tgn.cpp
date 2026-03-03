import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_TGUF_BIN = _SCRIPT_DIR.parent / "build" / "tools" / "tguf_cli" / "tguf_cli"

if not _TGUF_BIN.exists():
    raise ValueError(f"TGUF CLI Binary not found at {_TGUF_BIN}")


class TGUFStreamer:
    def __init__(
        self,
        out_path: Path,
        n_edges: int,
        m_dim: int,
        n_neg: int,
        n_labels: int,
        l_dim: int,
        val_start: int = 0,
        test_start: int = 0,
    ) -> None:
        self.proc = subprocess.Popen(
            [_TGUF_BIN, str(out_path)], stdin=subprocess.PIPE, stderr=sys.stderr
        )
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open pipe to tguf_cli")
        self.cpp_buffer = self.proc.stdin

        # Send header
        header = struct.pack(
            "7Q", n_edges, m_dim, n_neg, n_labels, l_dim, val_start, test_start
        )
        self.cpp_buffer.write(header)

        self.m_dim = m_dim
        self.n_neg = n_neg
        self.l_dim = l_dim

    def stream_edge_batch(self, src, dst, ts, msg, negs):
        batch_size = len(src)
        self.cpp_buffer.write(b"E")
        self.cpp_buffer.write(struct.pack("Q", batch_size))

        self._write_array(src, np.int64)
        self._write_array(dst, np.int64)
        self._write_array(ts, np.int64)
        if self.m_dim > 0:
            self._write_array(msg, np.float32)
        if self.n_neg > 0:
            self._write_array(negs, np.int64)

    def stream_label_batch(self, ts, nodes, labels):
        batch_size = len(ts)
        self.cpp_buffer.write(b"L")
        self.cpp_buffer.write(struct.pack("Q", batch_size))

        self._write_array(nodes, np.int64)
        self._write_array(ts, np.int64)
        self._write_array(labels, np.float32)

    def finalize(self):
        if self.cpp_buffer:
            self.cpp_buffer.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"CLI failed with exit code {ret}")

    def _write_array(self, x: np.ndarray, dtype) -> None:  # type: ignore
        if x.dtype != dtype:
            x = x.astype(dtype)
        x = np.ascontiguousarray(x)
        self.cpp_buffer.write(x.data)
