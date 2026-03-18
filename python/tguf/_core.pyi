from typing import Any, Optional, Union

import numpy as np
import torch

# Typings for the ndarray inputs (DLPack/Buffer Protocol compatible)
ArrayLike = Union[np.ndarray, torch.Tensor, Any]

class TGUFSchema:
    """Metadata defining the layout of a TGUF file."""

    path: str
    edge_capacity: int
    msg_dim: int
    label_dim: int
    node_feat_capacity: int
    node_feat_dim: int
    label_capacity: int
    negatives_start_e_id: int
    negatives_per_edge: int
    val_start: Optional[int]
    test_start: Optional[int]

    def __init__(
        self,
        path: str,
        edge_capacity: Optional[int] = None,
        msg_dim: Optional[int] = None,
        label_dim: Optional[int] = None,
        node_feat_capacity: Optional[int] = None,
        node_feat_dim: Optional[int] = None,
        label_capacity: Optional[int] = None,
        negatives_start_e_id: Optional[int] = None,
        negatives_per_edge: Optional[int] = None,
        val_start: Optional[int] = None,
        test_start: Optional[int] = None,
    ) -> None: ...

class Batch:
    """Container for temporal edge data."""

    src: torch.Tensor
    dst: torch.Tensor
    time: torch.Tensor
    msg: torch.Tensor
    neg_dst: Optional[torch.Tensor]

    def __init__(
        self,
        src: ArrayLike,
        dst: ArrayLike,
        time: ArrayLike,
        msg: ArrayLike,
        neg_dst: Optional[ArrayLike] = None,
    ) -> None: ...

class TGUFBuilder:
    """High-performance writer for creating TGUF datasets on disk."""

    def __init__(self, schema: TGUFSchema) -> None:
        """Initializes the builder with a specific TGUFSchema."""
        ...

    def append_edges(self, batch: Batch) -> None:
        """Appends a batch of edges to the persistent store."""
        ...

    def append_labels(
        self, n_id: ArrayLike, time: ArrayLike, target: ArrayLike
    ) -> None:
        """Appends a batch of label events to the persistent store."""
        ...

    def append_node_feats(self, n_id: ArrayLike, node_feat: ArrayLike) -> None:
        """Appends a batch of static node features to the persistent store."""
        ...

    def finalize(self) -> None:
        """Finalizes the .tguf file, writing headers and flushing buffers."""
        ...
