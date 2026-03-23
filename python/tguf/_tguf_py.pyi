"""
High-performance temporal graph learning primitives exposed from C++ via nanobind.

This module provides core data structures for constructing and writing
Temporal Graph Unified Format (TGUF) datasets.
"""

import enum
from typing import Annotated

import numpy
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

    def __init__(
        self,
        path: str,
        edge_capacity: int | None = None,
        msg_dim: int | None = None,
        label_dim: int | None = None,
        node_feat_capacity: int | None = None,
        node_feat_dim: int | None = None,
        label_capacity: int | None = None,
        negatives_start_e_id: int | None = None,
        negatives_per_edge: int | None = None,
        val_start: int | None = None,
        test_start: int | None = None,
    ) -> None: ...
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

    def __init__(
        self,
        src: NDArray,
        dst: NDArray,
        time: NDArray,
        msg: NDArray,
        neg_dst: NDArray | None = None,
    ) -> None: ...
    @property
    def src(self) -> Annotated[NDArray[numpy.int64], dict(shape=(1))]:
        """Source node IDs"""

    @property
    def dst(self) -> Annotated[NDArray[numpy.int64], dict(shape=(1))]:
        """Destination node IDs"""

    @property
    def time(self) -> Annotated[NDArray[numpy.int64], dict(shape=(1))]:
        """Edge Timestamps"""

    @property
    def msg(self) -> Annotated[NDArray[numpy.float32], dict(shape=(2))]:
        """Edge Features"""

    @property
    def neg_dst(self) -> object:
        """Optional negative destinations for link prediction"""

class LabelEvent:
    """
    Container for a label event at a single point in time.

    This structure represents node-centric targets (classification or regression)
    occurring at a specific timestamp in the temporal graph.

    Args:
        n_id (ndarray):
            Node IDs associated with the labels, shape [B], dtype=int64.
        target (ndarray):
            Label target values, shape [B, label_dim], dtype=float32.
    """

    def __init__(self, n_id: NDArray, target: NDArray) -> None: ...
    @property
    def n_id(self) -> Annotated[NDArray[numpy.int64], dict(shape=(1))]:
        """Node IDs associated with this label event."""

    @property
    def target(self) -> Annotated[NDArray[numpy.float32], dict(shape=(2))]:
        """Label target values (features/classes)"""

class NegStrategy(enum.Enum):
    """Negative sampling strategies for batch retrieval."""

    Random = 1
    """Samples one random negative node per edge."""

    PreComputed = 2
    """Uses fixed negatives stored in TGUF (for eval)."""

Random: NegStrategy = NegStrategy.Random

PreComputed: NegStrategy = NegStrategy.PreComputed

class IndexRange:
    """A contiguous slice of the graph data."""

    def __init__(self, arg0: int, arg1: int, /) -> None: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def size(self) -> int: ...

class TGStore:
    """
    Abstract interface for temporal graph storage.

    Implementations can be purely in-memory or memory-mapped TGUF files.
    Use :meth:`from_memory` or :meth:`from_tguf` to instantiate.
    """

    @staticmethod
    def from_memory(
        edges: Batch,
        node_feats: NDArray | None = None,
        label_n_id: NDArray | None = None,
        label_time: NDArray | None = None,
        label_target: NDArray | None = None,
        val_start: int | None = None,
        test_start: int | None = None,
    ) -> TGStore:
        """Create a high-speed, purely RAM-based store."""

    @staticmethod
    def from_tguf(
        path: str, val_start: int | None = None, test_start: int | None = None
    ) -> TGStore:
        """Create a memory-mapped store from a TGUF file."""

    @property
    def edge_count(self) -> int: ...
    @property
    def node_count(self) -> int: ...
    @property
    def label_count(self) -> int: ...
    @property
    def msg_dim(self) -> int: ...
    @property
    def label_dim(self) -> int: ...
    @property
    def node_feat_dim(self) -> int: ...
    @property
    def train_split(self) -> IndexRange: ...
    @property
    def val_split(self) -> IndexRange: ...
    @property
    def test_split(self) -> IndexRange: ...
    @property
    def train_label_split(self) -> IndexRange: ...
    @property
    def val_label_split(self) -> IndexRange: ...
    @property
    def test_label_split(self) -> IndexRange: ...
    def get_batch(
        self, start: int, size: int, strategy: NegStrategy = NegStrategy.None_
    ) -> Batch:
        """Retrieve a zero-copy slice of the graph interaction data."""

    def gather_timestamps(
        self, e_id: NDArray
    ) -> Annotated[NDArray[numpy.int64], dict(shape=(1))]:
        """Vectorized gather of edge timestamps."""

    def gather_msgs(
        self, e_id: NDArray
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(2))]:
        """Vectorized gather of edge features (messages)."""

    def gather_node_feats(
        self, n_id: NDArray
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(2))]:
        """Vectorized gather of static node features."""

    def get_edge_cutoff_for_label_event(self, l_id: int) -> int:
        """
        Retrieves the maximum edge_id that can be safely processed before a label.
        """

    def get_label_event(self, l_id: int) -> LabelEvent:
        """Retrieve a specific label event."""

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
