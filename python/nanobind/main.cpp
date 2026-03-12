#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <torch/torch.h>

#include <cstdint>

#include "tgn.h"

namespace nb = nanobind;

// This takes any Python object supporting DLPack/Buffer Protocol
torch::Tensor tensor_view(nb::ndarray<> array, torch::ScalarType type) {
  std::vector<std::int64_t> shape;
  for (auto i = 0; i < array.ndim(); ++i) {
    shape.push_back(array.shape(i));
  }
  return torch::from_blob(array.data(), shape,
                          torch::TensorOptions().dtype(type));
}

NB_MODULE(_core, m) {
  nb::class_<tgn::TGUFSchema>(m, "TGUFSchema")
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

  nb::class_<tgn::Batch>(m, "Batch")
      .def(
          "__init__",
          [](tgn::Batch *self, nb::ndarray<> src, nb::ndarray<> dst,
             nb::ndarray<> time, nb::ndarray<> msg,
             std::optional<nb::ndarray<>> neg_dst) {
            new (self) tgn::Batch{
                tensor_view(src, torch::kLong), tensor_view(dst, torch::kLong),
                tensor_view(time, torch::kLong),
                tensor_view(msg, torch::kFloat),
                neg_dst
                    ? std::make_optional(tensor_view(*neg_dst, torch::kLong))
                    : std::nullopt};
          },
          nb::arg("src"), nb::arg("dst"), nb::arg("time"), nb::arg("msg"),
          nb::arg("neg_dst") = nb::none());

  nb::class_<tgn::TGUFBuilder>(m, "TGUFBuilder")
      .def(nb::init<const tgn::TGUFSchema &>(), nb::arg("schema"))

      .def(
          "append_edges",
          [](const tgn::TGUFBuilder &self, const tgn::Batch &batch) {
            nb::gil_scoped_release release;
            self.append_edges(batch);
          },
          nb::arg("batch"))

      .def(
          "append_labels",
          [](const tgn::TGUFBuilder &self, nb::ndarray<> n_id,
             nb::ndarray<> time, nb::ndarray<> target) {
            nb::gil_scoped_release release;

            self.append_labels(tensor_view(n_id, torch::kLong),
                               tensor_view(time, torch::kLong),
                               tensor_view(target, torch::kFloat));
          },
          nb::arg("n_id"), nb::arg("time"), nb::arg("target"))

      .def(
          "append_node_feats",
          [](const tgn::TGUFBuilder &self, nb::ndarray<> n_id,
             nb::ndarray<> node_feat) {
            nb::gil_scoped_release release;

            self.append_node_feats(tensor_view(n_id, torch::kLong),
                                   tensor_view(node_feat, torch::kFloat));
          },
          nb::arg("n_id"), nb::arg("node_feat"))

      .def("finalize", [](tgn::TGUFBuilder &self) {
        nb::gil_scoped_release release;
        self.finalize();
      });
}
