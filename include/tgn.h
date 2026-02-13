#include <ATen/ops/_unique.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/unique_consecutive.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/rnn.h>
#include <torch/nn/options/linear.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "_recency_sampler.h"
#include "_scatter_ops.h"

namespace tgn {

using detail::LastNeighborLoader;
using detail::scatter_add;
using detail::scatter_argmax;
using detail::scatter_max;
using detail::scatter_softmax;

// Networks params
constexpr std::size_t EMBEDDING_DIM = 100;
constexpr std::size_t MEMORY_DIM = 100;
constexpr std::size_t TIME_DIM = 100;
constexpr std::size_t NUM_HEADS = 2;
constexpr std::size_t NUM_NBRS = 10;
constexpr float DROPOUT = 0.1;

// Data params
constexpr std::size_t NUM_NODES = 1000;
constexpr std::size_t MSG_DIM = 7;

struct TimeEncoderImpl : torch::nn::Module {
  explicit TimeEncoderImpl(std::size_t out_channels) {
    lin = register_module("lin", torch::nn::Linear(1, out_channels));
  }

  torch::Tensor forward(torch::Tensor t) {
    return lin->forward(t.view({-1, 1})).cos();
  }

 private:
  torch::nn::Linear lin{nullptr};
};
TORCH_MODULE(TimeEncoder);

struct TransformerConvImpl : torch::nn::Module {
  TransformerConvImpl(std::size_t in_channels, std::size_t out_channels,
                      std::size_t edge_dim, std::size_t heads,
                      float dropout = 0.0)
      : dropout_(dropout), out_channels_(out_channels), heads_(heads) {
    w_k = register_module("w_k",
                          torch::nn::Linear(in_channels, heads * out_channels));
    w_q = register_module("w_q",
                          torch::nn::Linear(in_channels, heads * out_channels));
    w_v = register_module("w_v",
                          torch::nn::Linear(in_channels, heads * out_channels));
    w_skip = register_module(
        "w_skip", torch::nn::Linear(in_channels, heads * out_channels));
    w_e = register_module(
        "w_e", torch::nn::Linear(
                   torch::nn::LinearOptions(edge_dim, heads * out_channels)
                       .bias(false)));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index,
                        torch::Tensor edge_attr) {
    const auto num_nodes = x.size(0);
    const auto j = edge_index[0];  // j is the sender
    const auto i = edge_index[1];  // i is the receiver

    const auto q = w_q->forward(x).view({-1, heads_, out_channels_});
    const auto k = w_k->forward(x).view({-1, heads_, out_channels_});
    const auto v = w_v->forward(x).view({-1, heads_, out_channels_});
    const auto e = w_e->forward(edge_attr).view({-1, heads_, out_channels_});

    // Message Function
    const auto k_j = k.index_select(0, j) + e;
    const auto q_i = q.index_select(0, i);
    auto a =
        (q_i * k_j).sum(-1) / std::sqrt(static_cast<double>(out_channels_));

    a = scatter_softmax(a, /* index*/ i, /* dim_size */ w_q->weight.size(0));
    a = torch::dropout(a, dropout_, is_training());

    auto out = (v.index_select(0, j) + e) * a.view({-1, heads_, 1});
    out = scatter_add(out, /* index */ i, /*dim_size*/ w_q->weight.size(0));
    out = out.view({num_nodes, heads_ * out_channels_});
    return out + w_skip->forward(x);
  }

 private:
  torch::nn::Linear w_k{nullptr}, w_q{nullptr}, w_v{nullptr}, w_e{nullptr},
      w_skip{nullptr};
  float dropout_{};
  std::size_t out_channels_{};
  std::size_t heads_{};
};
TORCH_MODULE(TransformerConv);

struct TGNMemoryImpl : torch::nn::Module {
  explicit TGNMemoryImpl(TimeEncoder time_encoder)
      : time_encoder_(time_encoder),
        num_nodes_(NUM_NODES),
        memory_(torch::empty({NUM_NODES, MEMORY_DIM})),
        last_update_(torch::empty({NUM_NODES},
                                  torch::TensorOptions().dtype(torch::kLong))),
        assoc_(torch::empty({NUM_NODES},
                            torch::TensorOptions().dtype(torch::kLong))) {
    register_buffer("memory_", memory_);
    register_buffer("last_update_", last_update_);
    register_buffer("assoc_", assoc_);

    // since our identity msg is cat(mem[src], mem[dst], raw_msg, t_enc)
    constexpr auto cell_dim = MEMORY_DIM + MEMORY_DIM + MSG_DIM + TIME_DIM;
    gru_ = register_module("gru_", torch::nn::GRUCell(cell_dim, MEMORY_DIM));

    src_store_.resize(num_nodes_);
    dst_store_.resize(num_nodes_);
    reset_state();
  }

  auto reset_state() -> void {
    memory_.zero_();
    last_update_.zero_();
    reset_msg_store();
  }

  auto detach() -> void { memory_.detach_(); }

  auto forward(torch::Tensor n_id) -> std::tuple<torch::Tensor, torch::Tensor> {
    return is_training() ? get_updated_memory(n_id)
                         : std::make_pair(memory_.index_select(0, n_id),
                                          last_update_.index_select(0, n_id));
  }

  auto update_state(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                    torch::Tensor raw_msg) -> void {
    const auto [n_id, _] = at::_unique(torch::cat({src, dst}));

    if (is_training()) {
      update_memory(n_id);
      update_msg_store(src, dst, t, raw_msg, true);
      update_msg_store(dst, src, t, raw_msg, false);
    } else {
      update_msg_store(src, dst, t, raw_msg, true);
      update_msg_store(dst, src, t, raw_msg, false);
      update_memory(n_id);
    }
  }

  auto train(bool mode = true) -> void override {
    if (is_training() && !mode) {
      // Flush message store in case we just entered eval mode.
      update_memory(torch::arange(num_nodes_));
      reset_msg_store();
    }
    torch::nn::Module::train(mode);
  }

 private:
  auto reset_msg_store() -> void {
    // Message store format: (src, dst, t, msg)
    const auto i =
        memory_.new_empty(0, torch::TensorOptions().dtype(torch::kLong));
    const auto msg = memory_.new_empty({0, MSG_DIM});

    for (auto j = 0; j < num_nodes_; ++j) {
      src_store_[j] = std::make_tuple(i, i, i, msg);
      dst_store_[j] = std::make_tuple(i, i, i, msg);
    }
  }

  auto update_memory(torch::Tensor n_id) -> void {
    auto [memory_nid, last_update_nid] = get_updated_memory(n_id);
    memory_.index_put_({n_id}, memory_nid);
    last_update_.index_put_({n_id}, last_update_nid);
  }

  auto get_updated_memory(torch::Tensor n_id)
      -> std::tuple<torch::Tensor, torch::Tensor> {
    assoc_.index_put_({n_id}, torch::arange(n_id.size(0)));

    // Compute messages (src -> dst), then (dst -> src).
    const auto [msg_s, t_s, src_s] = compute_msg(n_id, true);
    const auto [msg_d, t_d, src_d] = compute_msg(n_id, false);

    // Aggregate messages.
    const auto idx = torch::cat({src_s, src_d}, 0);
    const auto msg = torch::cat({msg_s, msg_d}, 0);
    const auto t = torch::cat({t_s, t_d}, 0);
    const auto aggr =
        last_aggr(msg, assoc_.index_select(0, idx), t, n_id.size(0));

    // Get local copy of updated memory, and then last_update.
    auto updated_memory = gru_->forward(aggr, memory_.index_select(0, n_id));
    auto updated_last_update = scatter_max(t, idx, last_update_.size(0));
    updated_last_update = updated_last_update.index_select(0, n_id);
    return {updated_memory, updated_last_update};
  }

  auto update_msg_store(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                        torch::Tensor raw_msg, bool is_src_store) -> void {
    // Group interactions by node ID
    const auto [n_id_sorted, perm] = src.sort();
    const auto [unique_nid, _, count] = torch::unique_consecutive(
        n_id_sorted, /*return_inverse=*/true, /*return_counts=*/true);

    // Convert count tensor to a C++ vector for split_with_sizes
    std::vector<std::int64_t> sizes(count.numel());
    auto count_acc = count.accessor<int64_t, 1>();
    for (std::int64_t i = 0; i < count.numel(); ++i) {
      sizes[i] = count_acc[i];
    }

    // Reorder all data based on the sorted node IDs and split them
    const auto src_s = src.index_select(0, perm).split_with_sizes(sizes);
    const auto dst_s = dst.index_select(0, perm).split_with_sizes(sizes);
    const auto t_s = t.index_select(0, perm).split_with_sizes(sizes);
    const auto msg_s = raw_msg.index_select(0, perm).split_with_sizes(sizes);

    auto& store = is_src_store ? src_store_ : dst_store_;

    for (std::int64_t i = 0; i < unique_nid.size(0); ++i) {
      const auto key = unique_nid[i].item<std::int64_t>();
      const auto value = std::make_tuple(src_s[i], dst_s[i], t_s[i], msg_s[i]);
      store[key] = value;
    }
  }

  auto compute_msg(torch::Tensor n_id, bool is_src_store)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    // Gather stored messages
    std::vector<torch::Tensor> src_list;
    std::vector<torch::Tensor> dst_list;
    std::vector<torch::Tensor> t_list;
    std::vector<torch::Tensor> raw_msg_list;

    const auto& store = is_src_store ? src_store_ : dst_store_;
    auto n_acc = n_id.accessor<std::int64_t, 1>();
    for (std::int64_t i = 0; i < n_id.numel(); ++i) {
      auto node = n_acc[i];

      const auto& data = store[node];
      src_list.push_back(std::get<0>(data));
      dst_list.push_back(std::get<1>(data));
      t_list.push_back(std::get<2>(data));
      raw_msg_list.push_back(std::get<3>(data));
    }

    const auto src = torch::cat(src_list, 0);
    const auto dst = torch::cat(dst_list, 0);
    const auto t = torch::cat(t_list, 0);
    const auto raw_msg = torch::cat(raw_msg_list, 0);

    // Compute msg components
    const auto rel_t = t - last_update_.index_select(0, src);
    const auto rel_t_z = time_encoder_->forward(rel_t.to(raw_msg.dtype()));
    const auto mem_src = memory_.index_select(0, src);
    const auto mem_dst = memory_.index_select(0, dst);

    // Final message (identity aggr)
    const auto msg = torch::cat({mem_src, mem_dst, raw_msg, rel_t_z}, 1);

    return std::make_tuple(msg, t, src);
  }

  auto last_aggr(torch::Tensor msg, torch::Tensor index, torch::Tensor t,
                 int dim_size) -> torch::Tensor {
    auto out = msg.new_zeros({dim_size, msg.size(-1)});

    // Number of messages is t.numel();
    if (t.numel()) {
      const auto argmax = scatter_argmax(t, index, dim_size);
      const auto mask = argmax < msg.size(0);  // Items with at least one entry
      out.index_put_({mask}, argmax.index({mask}));
    }

    return out;
  }

  std::size_t num_nodes_{};
  torch::Tensor memory_{};
  torch::Tensor last_update_{};
  torch::Tensor assoc_{};

  TimeEncoder time_encoder_{nullptr};
  torch::nn::GRUCell gru_{nullptr};

  std::vector<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      src_store_, dst_store_;
};
TORCH_MODULE(TGNMemory);

struct TGNImpl : torch::nn::Module {
  TGNImpl()
      : assoc_(torch::full({NUM_NODES}, -1,
                           torch::TensorOptions().dtype(torch::kLong))),
        nbr_loader_(NUM_NBRS, NUM_NODES) {
    time_encoder_ = register_module("time_encoder_", TimeEncoder(TIME_DIM));
    memory_ = register_module("memory_", TGNMemory(time_encoder_));
    conv_ = register_module(
        "conv_", TransformerConv(MEMORY_DIM, EMBEDDING_DIM / 2,
                                 MSG_DIM + TIME_DIM, NUM_HEADS, DROPOUT));
  }

  auto reset_state() -> void {
    memory_->reset_state();
    nbr_loader_.reset_state();
  }

  auto update_state(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                    torch::Tensor msg) -> void {
    memory_->update_state(src, dst, t, msg);
    nbr_loader_.insert(src, dst);
  }

  auto detach_memory() -> void { memory_->detach(); }

  torch::Tensor forward(torch::Tensor global_n_id) {
    const auto [n_id, edge_index, e_id] = nbr_loader_(global_n_id);
    const auto [x, last_update] = memory_(n_id);

    assoc_.index_put_({n_id}, torch::arange(n_id.size(0), assoc_.options()));

    if (e_id.numel() > 0) {
      // TODO(kuba): global ref to data
      const auto t = torch::rand({n_id.size(0)});             // data.t[e_id]
      const auto msg = torch::rand({n_id.size(0), MSG_DIM});  // data.msg[e_id]

      const auto rel_t = last_update.index_select(0, edge_index[0]) - t;
      const auto rel_t_z = time_encoder_->forward(rel_t);
      const auto edge_attr = torch::cat({rel_t_z, msg}, -1);

      z_cache_ = conv_(x, edge_index, edge_attr);
    } else {
      z_cache_ = x;
    }

    return z_cache_;
  }

  auto get_embeddings(torch::Tensor global_n_id) -> torch::Tensor {
    const auto local_indices = assoc_.index({global_n_id});
    return z_cache_.index_select(0, local_indices);
  }

 private:
  TimeEncoder time_encoder_{nullptr};
  TransformerConv conv_{nullptr};
  TGNMemory memory_{nullptr};
  LastNeighborLoader nbr_loader_;

  torch::Tensor z_cache_;
  torch::Tensor assoc_;
};
TORCH_MODULE(TGN);

}  // namespace tgn
