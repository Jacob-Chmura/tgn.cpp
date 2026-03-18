"""
High-performance temporal graph learning primitives exposed from C++ via nanobind.

This module provides core data structures for constructing and writing
Temporal Graph Unified Format (TGUF) datasets.
"""

from numpy.typing import NDArray


class TGUFSchema:
    """
    Metadata defining the layout of a TGUF dataset.

    This schema specifies dataset capacities, feature dimensions, and optional
    evaluation splits. It is required to initialize a :class:`TGUFBuilder`.

    Args:
        path (str):
            Path to the `.tguf` binary file.

        edge_capacity (int, optional):
            Maximum number of edges.

        msg_dim (int, optional):
            Dimension of edge features.

        label_dim (int, optional):
            Dimension of label targets.

        node_feat_capacity (int, optional):
            Maximum number of nodes with static features.

        node_feat_dim (int, optional):
            Dimension of node features.

        label_capacity (int, optional):
            Maximum number of label events.

        negatives_start_e_id (int, optional):
            Edge index where precomputed negatives begin (for evaluation).

        negatives_per_edge (int, optional):
            Number of negatives per edge.

        val_start (int, optional):
            Global edge index where validation split begins.

        test_start (int, optional):
            Global edge index where test split begins.

    Notes:
        If `val_start` or `test_start` are not provided, the dataset is treated
        as fully training unless overridden during loading.
    """

    def __init__(self, path: str, edge_capacity: int | None = None, msg_dim: int | None = None, label_dim: int | None = None, node_feat_capacity: int | None = None, node_feat_dim: int | None = None, label_capacity: int | None = None, negatives_start_e_id: int | None = None, negatives_per_edge: int | None = None, val_start: int | None = None, test_start: int | None = None) -> None: ...

    @property
    def path(self) -> str: ...

    @path.setter
    def path(self, arg: str, /) -> None: ...

    @property
    def edge_capacity(self) -> int: ...

    @edge_capacity.setter
    def edge_capacity(self, arg: int, /) -> None: ...

    @property
    def msg_dim(self) -> int: ...

    @msg_dim.setter
    def msg_dim(self, arg: int, /) -> None: ...

    @property
    def label_dim(self) -> int: ...

    @label_dim.setter
    def label_dim(self, arg: int, /) -> None: ...

    @property
    def node_feat_capacity(self) -> int: ...

    @node_feat_capacity.setter
    def node_feat_capacity(self, arg: int, /) -> None: ...

    @property
    def node_feat_dim(self) -> int: ...

    @node_feat_dim.setter
    def node_feat_dim(self, arg: int, /) -> None: ...

    @property
    def label_capacity(self) -> int: ...

    @label_capacity.setter
    def label_capacity(self, arg: int, /) -> None: ...

    @property
    def negatives_start_e_id(self) -> int: ...

    @negatives_start_e_id.setter
    def negatives_start_e_id(self, arg: int, /) -> None: ...

    @property
    def negatives_per_edge(self) -> int: ...

    @negatives_per_edge.setter
    def negatives_per_edge(self, arg: int, /) -> None: ...

    @property
    def val_start(self) -> int | None: ...

    @val_start.setter
    def val_start(self, arg: int | None) -> None: ...

    @property
    def test_start(self) -> int | None: ...

    @test_start.setter
    def test_start(self, arg: int | None) -> None: ...

class Batch:
    """
    Container for temporal edge data.

    This structure represents a batch of temporal interactions and is used
    as input to :meth:`TGUFBuilder.append_edges`.

    Args:
        src (ndarray):
            Source node IDs of shape [B], dtype=int64.

        dst (ndarray):
            Destination node IDs of shape [B], dtype=int64.

        time (ndarray):
            Timestamps of shape [B], dtype=int64.

        msg (ndarray):
            Edge features of shape [B, msg_dim], dtype=float32.

        neg_dst (ndarray, optional):
            Negative destination nodes for link prediction of shape
            [B, negatives_per_edge], dtype=int64.

    Notes:
        All inputs are converted to PyTorch tensors internally.

    See also:
        - :class:`TGUFBuilder`
    """

    def __init__(self, src: NDArray, dst: NDArray, time: NDArray, msg: NDArray, neg_dst: NDArray | None = None) -> None: ...

class TGUFBuilder:
    """
    High-performance writer for creating TGUF datasets on disk.

    Uses an internal buffering strategy to minimize disk I/O.

    Args:
        schema (TGUFSchema):
            Dataset schema defining layout and capacities.

    See also:
        - :class:`TGUFSchema`
        - :class:`Batch`
    """

    def __init__(self, schema: TGUFSchema) -> None: ...

    def append_edges(self, batch: Batch) -> None:
        """
        Append a batch of temporal edges to the dataset.

        Args:
            batch (Batch):
                A batch of temporal edge data.

        Notes:
            Releases the Python GIL during execution.
        """

    def append_labels(self, n_id: NDArray, time: NDArray, target: NDArray) -> None:
        """
        Append label events to the dataset.

        Args:
            n_id (ndarray):
                Node IDs of shape [B], dtype=int64.

            time (ndarray):
                Event timestamps of shape [B], dtype=int64.

            target (ndarray):
                Label targets of shape [B, label_dim], dtype=float32.

        Notes:
            Releases the Python GIL during execution.
        """

    def append_node_feats(self, n_id: NDArray, node_feat: NDArray) -> None:
        """
        Append static node features to the dataset.

        Args:
            n_id (ndarray):
                Node IDs of shape [N], dtype=int64.

            node_feat (ndarray):
                Node features of shape [N, node_feat_dim], dtype=float32.

        Notes:
            Releases the Python GIL during execution.
        """

    def finalize(self) -> None:
        """
        Finalize the dataset.

        Writes headers and flushes all buffered data to disk.

        Notes:
            Must be called after all data has been appended.
            Releases the Python GIL during execution.
        """
