#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tguf.h"

namespace nb = nanobind;

namespace {
// This takes any Python object supporting DLPack/Buffer Protocol
torch::Tensor tensor_view(const nb::ndarray<> &array, torch::ScalarType type) {
  std::vector<std::int64_t> shape;
  for (auto i = 0; i < array.ndim(); ++i) {
    shape.push_back(array.shape(i));
  }
  // TODO(kuba): Should have a legit zero-copy API
  return torch::from_blob(array.data(), shape,
                          torch::TensorOptions().dtype(type))
      .clone();
}

NB_MODULE(_tguf_py, m) {
  m.doc() = R"doc(
    High-performance temporal graph learning primitives exposed from C++ via nanobind.

    This module provides core data structures for constructing and writing
    Temporal Graph Unified Format (TGUF) datasets.

    )doc";

  nb::class_<tguf::TGUFSchema>(m, "TGUFSchema", R"doc(
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
          [](tguf::TGUFSchema *self, std::string path,
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
            new (self) tguf::TGUFSchema();
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

      .def_rw("path", &tguf::TGUFSchema::path)
      .def_rw("edge_capacity", &tguf::TGUFSchema::edge_capacity)
      .def_rw("msg_dim", &tguf::TGUFSchema::msg_dim)
      .def_rw("label_dim", &tguf::TGUFSchema::label_dim)
      .def_rw("node_feat_capacity", &tguf::TGUFSchema::node_feat_capacity)
      .def_rw("node_feat_dim", &tguf::TGUFSchema::node_feat_dim)
      .def_rw("label_capacity", &tguf::TGUFSchema::label_capacity)
      .def_rw("negatives_start_e_id", &tguf::TGUFSchema::negatives_start_e_id)
      .def_rw("negatives_per_edge", &tguf::TGUFSchema::negatives_per_edge)
      .def_rw("val_start", &tguf::TGUFSchema::val_start)
      .def_rw("test_start", &tguf::TGUFSchema::test_start);

  nb::class_<tguf::Batch>(m, "Batch", R"doc(
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
          [](tguf::Batch *self, nb::ndarray<> src, nb::ndarray<> dst,
             nb::ndarray<> time, nb::ndarray<> msg,
             std::optional<nb::ndarray<>> neg_dst) {
            new (self)
                tguf::Batch{.src = tensor_view(src, torch::kLong),
                            .dst = tensor_view(dst, torch::kLong),
                            .time = tensor_view(time, torch::kLong),
                            .msg = tensor_view(msg, torch::kFloat),
                            .neg_dst = neg_dst ? std::make_optional(tensor_view(
                                                     *neg_dst, torch::kLong))
                                               : std::nullopt};
          },
          nb::arg("src"), nb::arg("dst"), nb::arg("time"), nb::arg("msg"),
          nb::arg("neg_dst") = nb::none())

      .def_prop_ro(
          "src",
          [](tguf::Batch &b) {
            return nb::ndarray<nb::pytorch, std::int64_t, nb::shape<1>>(
                b.src.data_ptr<std::int64_t>(),
                {static_cast<std::size_t>(b.src.size(0))}, nb::handle());
          },
          "Source node IDs")
      .def_prop_ro(
          "dst",
          [](tguf::Batch &b) {
            return nb::ndarray<nb::pytorch, std::int64_t, nb::shape<1>>(
                b.dst.data_ptr<std::int64_t>(),
                {static_cast<std::size_t>(b.dst.size(0))}, nb::handle());
          },
          "Destination node IDs")
      .def_prop_ro(
          "time",
          [](tguf::Batch &b) {
            return nb::ndarray<nb::pytorch, std::int64_t, nb::shape<1>>(
                b.time.data_ptr<std::int64_t>(),
                {static_cast<std::size_t>(b.time.size(0))}, nb::handle());
          },
          "Edge Timestamps")
      .def_prop_ro(
          "msg",
          [](tguf::Batch &b) {
            return nb::ndarray<nb::pytorch, float, nb::shape<2>>(
                b.msg.data_ptr<float>(),
                {static_cast<std::size_t>(b.msg.size(0)),
                 static_cast<std::size_t>(b.msg.size(1))},
                nb::handle());
          },
          "Edge Features")
      .def_prop_ro(
          "neg_dst",
          [](tguf::Batch &b) -> nb::object {
            if (b.neg_dst.has_value()) {
              return nb::cast(
                  nb::ndarray<nb::pytorch, std::int64_t, nb::shape<2>>(
                      b.neg_dst->data_ptr<std::int64_t>(),
                      {static_cast<std::size_t>(b.neg_dst->size(0)),
                       static_cast<std::size_t>(b.neg_dst->size(1))},
                      nb::handle()));
            } else {
              return nb::none();
            }
          },
          "Optional negative destinations for link prediction");

  nb::enum_<tguf::TGStore::NegStrategy>(
      m, "NegStrategy", "Negative sampling strategies for batch retrieval.")
      .value("None_", tguf::TGStore::NegStrategy::None,
             "No negatives (inference or node-level tasks).")
      .value("Random", tguf::TGStore::NegStrategy::Random,
             "Samples one random negative node per edge.")
      .value("PreComputed", tguf::TGStore::NegStrategy::PreComputed,
             "Uses fixed negatives stored in TGUF (for eval).")
      .export_values();

  nb::class_<tguf::TGStore::IndexRange>(m, "IndexRange",
                                        "A contiguous slice of the graph data.")
      .def(nb::init<std::size_t, std::size_t>())
      .def_prop_ro("start", &tguf::TGStore::IndexRange::start)
      .def_prop_ro("end", &tguf::TGStore::IndexRange::end)
      .def_prop_ro("size", &tguf::TGStore::IndexRange::size);

  nb::class_<tguf::TGStore>(m, "TGStore", R"doc(
Abstract interface for temporal graph storage.

Implementations can be purely in-memory or memory-mapped TGUF files.
Use :meth:`from_memory` or :meth:`from_tguf` to instantiate.
)doc")
      .def_static("from_memory", &tguf::TGStore::from_memory, nb::arg("edges"),
                  nb::arg("node_feats") = nb::none(),
                  nb::arg("label_n_id") = nb::none(),
                  nb::arg("label_time") = nb::none(),
                  nb::arg("label_target") = nb::none(),
                  nb::arg("val_start") = nb::none(),
                  nb::arg("test_start") = nb::none(),
                  "Create a high-speed, purely RAM-based store.")

      .def_static("from_tguf", &tguf::TGStore::from_tguf, nb::arg("path"),
                  nb::arg("val_start") = nb::none(),
                  nb::arg("test_start") = nb::none(),
                  "Create a memory-mapped store from a TGUF file.")

      .def_prop_ro("edge_count", &tguf::TGStore::edge_count)
      .def_prop_ro("node_count", &tguf::TGStore::node_count)
      .def_prop_ro("msg_dim", &tguf::TGStore::msg_dim)
      .def_prop_ro("label_dim", &tguf::TGStore::label_dim)
      .def_prop_ro("node_feat_dim", &tguf::TGStore::node_feat_dim)

      .def_prop_ro("train_split", &tguf::TGStore::train_split)
      .def_prop_ro("val_split", &tguf::TGStore::val_split)
      .def_prop_ro("test_split", &tguf::TGStore::test_split)
      .def_prop_ro("train_label_split", &tguf::TGStore::train_label_split)
      .def_prop_ro("val_label_split", &tguf::TGStore::val_label_split)
      .def_prop_ro("test_label_split", &tguf::TGStore::test_label_split)

      .def(
          "get_batch",
          [](const tguf::TGStore &self, std::size_t start, std::size_t size,
             tguf::TGStore::NegStrategy strategy) {
            nb::gil_scoped_release release;
            return self.get_batch(start, size, strategy);
          },
          nb::arg("start"), nb::arg("size"),
          nb::arg("strategy") = tguf::TGStore::NegStrategy::None,
          "Retrieve a zero-copy slice of the graph interaction data.")

      .def(
          "gather_timestamps",
          [](const tguf::TGStore &self, nb::ndarray<> e_id) {
            nb::gil_scoped_release release;
            auto res = self.gather_timestamps(tensor_view(e_id, torch::kLong));
            return nb::ndarray<nb::pytorch, std::int64_t, nb::shape<1>>(
                res.data_ptr<std::int64_t>(),
                {static_cast<std::size_t>(res.size(0))}, nb::handle());
          },
          nb::arg("e_id"), "Vectorized gather of edge timestamps.")

      .def(
          "gather_msgs",
          [](const tguf::TGStore &self, nb::ndarray<> e_id) {
            nb::gil_scoped_release release;
            auto res = self.gather_msgs(tensor_view(e_id, torch::kLong));
            return nb::ndarray<nb::pytorch, float, nb::shape<2>>(
                res.data_ptr<float>(),
                {static_cast<std::size_t>(res.size(0)),
                 static_cast<std::size_t>(res.size(1))},
                nb::handle());
          },
          nb::arg("e_id"), "Vectorized gather of edge features (messages).")

      .def(
          "gather_node_feats",
          [](const tguf::TGStore &self, nb::ndarray<> n_id) {
            nb::gil_scoped_release release;
            auto res = self.gather_node_feats(tensor_view(n_id, torch::kLong));
            return nb::ndarray<nb::pytorch, float, nb::shape<2>>(
                res.data_ptr<float>(),
                {static_cast<std::size_t>(res.size(0)),
                 static_cast<std::size_t>(res.size(1))},
                nb::handle());
          },
          nb::arg("n_id"), "Vectorized gather of static node features.")

      .def("get_edge_cutoff_for_label_event",
           &tguf::TGStore::get_edge_cutoff_for_label_event, nb::arg("l_id"),
           "Retrieves the maximum edge_id that can be safely processed before "
           "a label.")

      .def(
          "get_label_event",
          [](const tguf::TGStore &self, std::size_t l_id) {
            nb::gil_scoped_release release;
            return self.get_label_event(l_id);
          },
          nb::arg("l_id"), "Retrieve a specific label event.");

  nb::class_<tguf::LabelEvent>(m, "LabelEvent", R"doc(
Container for a label event at a single point in time.

This structure represents node-centric targets (classification or regression)
occurring at a specific timestamp in the temporal graph.

Args:
    n_id (ndarray):
        Node IDs associated with the labels, shape [B], dtype=int64.
    target (ndarray):
        Label target values, shape [B, label_dim], dtype=float32.
)doc")
      .def(
          "__init__",
          [](tguf::LabelEvent *self, nb::ndarray<> n_id, nb::ndarray<> target) {
            new (self)
                tguf::LabelEvent{.n_id = tensor_view(n_id, torch::kLong),
                                 .target = tensor_view(target, torch::kFloat)};
          },
          nb::arg("n_id"), nb::arg("target"))

      .def_prop_ro(
          "n_id",
          [](tguf::LabelEvent &le) {
            return nb::ndarray<nb::pytorch, std::int64_t, nb::shape<1>>(
                le.n_id.data_ptr<std::int64_t>(),
                {static_cast<std::size_t>(le.n_id.size(0))}, nb::handle());
          },
          "Node IDs associated with this label event.")
      .def_prop_ro(
          "target",
          [](tguf::LabelEvent &le) {
            return nb::ndarray<nb::pytorch, std::int64_t, nb::shape<1>>(
                le.target.data_ptr<std::int64_t>(),
                {static_cast<std::size_t>(le.target.size(0))}, nb::handle());
          },
          "Label target values (features/classes)");

  nb::class_<tguf::TGUFBuilder>(m, "TGUFBuilder",
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
      .def(nb::init<const tguf::TGUFSchema &>(), nb::arg("schema"))

      .def(
          "append_edges",
          [](const tguf::TGUFBuilder &self, const tguf::Batch &batch) {
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
          [](const tguf::TGUFBuilder &self, nb::ndarray<> n_id,
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
          [](const tguf::TGUFBuilder &self, nb::ndarray<> n_id,
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
          [](tguf::TGUFBuilder &self) {
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
