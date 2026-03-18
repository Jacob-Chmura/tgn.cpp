#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tgn.h"

namespace nb = nanobind;

namespace {
// This takes any Python object supporting DLPack/Buffer Protocol
torch::Tensor tensor_view(const nb::ndarray<> &array, torch::ScalarType type) {
  std::vector<std::int64_t> shape;
  for (auto i = 0; i < array.ndim(); ++i) {
    shape.push_back(array.shape(i));
  }
  return torch::from_blob(array.data(), shape,
                          torch::TensorOptions().dtype(type))
      .clone();
}

NB_MODULE(_tguf, m) {
  m.doc() = R"doc(
    High-performance temporal graph learning primitives exposed from C++ via nanobind.

    This module provides core data structures for constructing and writing
    Temporal Graph Unified Format (TGUF) datasets.

    )doc";
  nb::class_<tgn::TGUFSchema>(m, "TGUFSchema", R"doc(
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
)doc")
      .def(
          "__init__",
          [](tgn::TGUFSchema *self, std::string path,
             std::optional<std::size_t> edge_capacity,
             std::optional<std::size_t> msg_dim,
             std::optional<std::size_t> label_dim,
             std::optional<std::size_t> node_feat_capacity,
             std::optional<std::size_t> node_feat_dim,
             std::optional<std::size_t> label_capacity,
             std::optional<std::size_t> negatives_start_e_id,
             std::optional<std::size_t> negatives_per_edge,
             std::optional<std::size_t> val_start,
             std::optional<std::size_t> test_start) {
            new (self) tgn::TGUFSchema();
            self->path = std::move(path);
            self->edge_capacity = edge_capacity.value_or(0);
            self->msg_dim = msg_dim.value_or(0);
            self->node_feat_capacity = node_feat_capacity.value_or(0);
            self->node_feat_dim = node_feat_dim.value_or(0);
            self->label_capacity = label_capacity.value_or(0);
            self->label_dim = label_dim.value_or(0);
            self->negatives_start_e_id = negatives_start_e_id.value_or(0);
            self->negatives_per_edge = negatives_per_edge.value_or(0);
            self->val_start = val_start;
            self->test_start = test_start;
          },
          nb::arg("path"), nb::arg("edge_capacity") = nb::none(),
          nb::arg("msg_dim") = nb::none(), nb::arg("label_dim") = nb::none(),
          nb::arg("node_feat_capacity") = nb::none(),
          nb::arg("node_feat_dim") = nb::none(),
          nb::arg("label_capacity") = nb::none(),
          nb::arg("negatives_start_e_id") = nb::none(),
          nb::arg("negatives_per_edge") = nb::none(),
          nb::arg("val_start") = nb::none(), nb::arg("test_start") = nb::none())

      .def_rw("path", &tgn::TGUFSchema::path)
      .def_rw("edge_capacity", &tgn::TGUFSchema::edge_capacity)
      .def_rw("msg_dim", &tgn::TGUFSchema::msg_dim)
      .def_rw("label_dim", &tgn::TGUFSchema::label_dim)
      .def_rw("node_feat_capacity", &tgn::TGUFSchema::node_feat_capacity)
      .def_rw("node_feat_dim", &tgn::TGUFSchema::node_feat_dim)
      .def_rw("label_capacity", &tgn::TGUFSchema::label_capacity)
      .def_rw("negatives_start_e_id", &tgn::TGUFSchema::negatives_start_e_id)
      .def_rw("negatives_per_edge", &tgn::TGUFSchema::negatives_per_edge)
      .def_rw("val_start", &tgn::TGUFSchema::val_start)
      .def_rw("test_start", &tgn::TGUFSchema::test_start);

  nb::class_<tgn::Batch>(m, "Batch", R"doc(
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
)doc")
      .def(
          "__init__",
          [](tgn::Batch *self, nb::ndarray<> src, nb::ndarray<> dst,
             nb::ndarray<> time, nb::ndarray<> msg,
             std::optional<nb::ndarray<>> neg_dst) {
            new (self)
                tgn::Batch{.src = tensor_view(src, torch::kLong),
                           .dst = tensor_view(dst, torch::kLong),
                           .time = tensor_view(time, torch::kLong),
                           .msg = tensor_view(msg, torch::kFloat),
                           .neg_dst = neg_dst ? std::make_optional(tensor_view(
                                                    *neg_dst, torch::kLong))
                                              : std::nullopt};
          },
          nb::arg("src"), nb::arg("dst"), nb::arg("time"), nb::arg("msg"),
          nb::arg("neg_dst") = nb::none());

  nb::class_<tgn::TGUFBuilder>(m, "TGUFBuilder",
                               R"doc(
High-performance writer for creating TGUF datasets on disk.

Uses an internal buffering strategy to minimize disk I/O.

Args:
    schema (TGUFSchema):
        Dataset schema defining layout and capacities.

See also:
    - :class:`TGUFSchema`
    - :class:`Batch`
)doc")
      .def(nb::init<const tgn::TGUFSchema &>(), nb::arg("schema"))

      .def(
          "append_edges",
          [](const tgn::TGUFBuilder &self, const tgn::Batch &batch) {
            nb::gil_scoped_release release;
            self.append_edges(batch);
          },
          nb::arg("batch"),
          R"doc(
Append a batch of temporal edges to the dataset.

Args:
    batch (Batch):
        A batch of temporal edge data.

Notes:
    Releases the Python GIL during execution.
)doc")

      .def(
          "append_labels",
          [](const tgn::TGUFBuilder &self, nb::ndarray<> n_id,
             nb::ndarray<> time, nb::ndarray<> target) {
            nb::gil_scoped_release release;

            self.append_labels(tensor_view(n_id, torch::kLong),
                               tensor_view(time, torch::kLong),
                               tensor_view(target, torch::kFloat));
          },
          nb::arg("n_id"), nb::arg("time"), nb::arg("target"),
          R"doc(
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
)doc")

      .def(
          "append_node_feats",
          [](const tgn::TGUFBuilder &self, nb::ndarray<> n_id,
             nb::ndarray<> node_feat) {
            nb::gil_scoped_release release;

            self.append_node_feats(tensor_view(n_id, torch::kLong),
                                   tensor_view(node_feat, torch::kFloat));
          },
          nb::arg("n_id"), nb::arg("node_feat"),
          R"doc(
Append static node features to the dataset.

Args:
    n_id (ndarray):
        Node IDs of shape [N], dtype=int64.

    node_feat (ndarray):
        Node features of shape [N, node_feat_dim], dtype=float32.

Notes:
    Releases the Python GIL during execution.
)doc")

      .def(
          "finalize",
          [](tgn::TGUFBuilder &self) {
            nb::gil_scoped_release release;
            self.finalize();
          },
          R"doc(
Finalize the dataset.

Writes headers and flushes all buffered data to disk.

Notes:
    Must be called after all data has been appended.
    Releases the Python GIL during execution.
)doc");
}
}  // namespace
