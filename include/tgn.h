#include <ATen/ops/_unique.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/unique_consecutive.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/rnn.h>
#include <torch/nn/options/linear.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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

struct TGNConfig {
  // Model parameters
  std::size_t embedding_dim = 100;
  std::size_t memory_dim = 100;
  std::size_t time_dim = 100;

  std::size_t num_heads = 2;
  std::size_t num_nbrs = 10;
  float dropout = 0.1;

  // Data parameters
  std::size_t num_nodes = 1000;
  std::size_t msg_dim = 7;
};

struct Batch {
  torch::Tensor src, dst, t, msg;
  torch::Tensor neg_dst;  // neg_dst should be std::optional<>
};

class TGStore {
 public:
  virtual ~TGStore() = default;
  virtual auto size() const -> std::size_t = 0;

  [[nodiscard]] virtual auto get_batch(std::size_t start,
                                       std::size_t batch_size) const
      -> Batch = 0;

  [[nodiscard]] virtual auto fetch_t(const torch::Tensor global_n_id) const
      -> torch::Tensor = 0;
  [[nodiscard]] virtual auto fetch_msg(const torch::Tensor global_n_id) const
      -> torch::Tensor = 0;
};

class SimpleTGStore final : public TGStore {
 public:
  explicit SimpleTGStore(const std::string& data_dir) {
    n_events_ = 100;
    msg_dim_ = 7;
  }

  auto size() const -> std::size_t override { return n_events_; }

  [[nodiscard]] auto get_batch(std::size_t start, std::size_t batch_size) const
      -> Batch override {
    const auto end = std::min(start + batch_size, n_events_);
    const auto current_batch_size = static_cast<std::int64_t>(end - start);
    return Batch{
        .src = torch::randint(0, 5, {current_batch_size}),
        .dst = torch::randint(0, 5, {current_batch_size}),
        .t = torch::arange(static_cast<std::int64_t>(start),
                           static_cast<std::int64_t>(end), torch::kLong),
        .msg = torch::zeros({current_batch_size, msg_dim_}),
        .neg_dst = torch::randint(0, 5, {current_batch_size}),
    };
  }

  [[nodiscard]] auto fetch_t(const torch::Tensor global_n_id) const
      -> torch::Tensor override {
    return torch::rand({global_n_id.size(0)});
  }

  [[nodiscard]] auto fetch_msg(const torch::Tensor global_n_id) const
      -> torch::Tensor override {
    return torch::rand({global_n_id.size(0), msg_dim_});
  }

 private:
  std::size_t n_events_{};
  std::int64_t msg_dim_{};
  torch::Tensor src_{}, dst_{}, t_{}, neg_dst_{};
};

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
      : dropout_(dropout),
        out_channels_(static_cast<std::int64_t>(out_channels)),
        heads_(static_cast<std::int64_t>(heads)) {
    const auto in_dim = static_cast<std::int64_t>(in_channels);
    const auto out_dim = heads_ * out_channels_;
    w_k = register_module("w_k", torch::nn::Linear(in_dim, out_dim));
    w_q = register_module("w_q", torch::nn::Linear(in_dim, out_dim));
    w_v = register_module("w_v", torch::nn::Linear(in_dim, out_dim));
    w_skip = register_module("w_skip", torch::nn::Linear(in_dim, out_dim));
    w_e = register_module(
        "w_e",
        torch::nn::Linear(torch::nn::LinearOptions(
                              static_cast<std::int64_t>(edge_dim), out_dim)
                              .bias(false)));
  }

  torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& edge_index,
                        const torch::Tensor& edge_attr) {
    // TODO(kuba): implement 2d scatter ops to avoid these huge flatten ops
    const auto B = x.size(0);
    const auto E = edge_index.size(1);
    const auto H = heads_;
    const auto C = out_channels_;
    const auto opts = edge_index.options();  // torch::LongTensor

    // Projections
    const auto q = w_q->forward(x).view({B, H, C});
    const auto k = w_k->forward(x).view({B, H, C});
    const auto v = w_v->forward(x).view({B, H, C});
    const auto e = w_e->forward(edge_attr).view({E, H, C});

    // Attention scores
    const auto src = edge_index[0];  // src is the sender
    const auto dst = edge_index[1];  // dst is the receiver

    const auto k_src = k.index_select(0, src) + e;
    const auto q_dst = q.index_select(0, dst);
    auto alpha = (q_dst * k_src).sum(-1) / std::sqrt(static_cast<double>(C));
    alpha = alpha.view(-1);  // flatten for 2-d scatter [E * H]

    // Scatter-softmax attention
    const auto H_offset = torch::arange(H, opts).repeat({E});
    auto scatter_idx = (dst.repeat_interleave(H) * H) + H_offset;

    alpha = scatter_softmax(alpha, scatter_idx, B * H);
    alpha = torch::dropout(alpha, dropout_, is_training());

    // Scatter-add message aggregation
    auto msgs = (v.index_select(0, src) + e) * alpha.view({E, H, 1});
    msgs = msgs.view(-1);  // flatten for 3-d scatter [E * H * C]

    const auto C_offset = torch::arange(C, opts).repeat({E * H});
    scatter_idx = (scatter_idx.repeat_interleave(C) * C) + C_offset;

    auto out = scatter_add(msgs, scatter_idx, B * H * C);
    out = out.view({B, H * C});

    return out + w_skip->forward(x);
  }

 private:
  torch::nn::Linear w_k{nullptr}, w_q{nullptr}, w_v{nullptr}, w_e{nullptr},
      w_skip{nullptr};
  float dropout_{};
  std::int64_t out_channels_{};
  std::int64_t heads_{};
};
TORCH_MODULE(TransformerConv);

struct TGNMemoryImpl : torch::nn::Module {
  explicit TGNMemoryImpl(const TGNConfig& cfg, TimeEncoder time_encoder)
      : msg_dim_(cfg.msg_dim),
        num_nodes_(cfg.num_nodes),
        memory_(torch::empty({static_cast<std::int64_t>(cfg.num_nodes),
                              static_cast<std::int64_t>(cfg.memory_dim)})),
        last_update_(torch::empty({static_cast<std::int64_t>(cfg.num_nodes)},
                                  torch::TensorOptions().dtype(torch::kLong))),
        assoc_(torch::empty({static_cast<std::int64_t>(cfg.num_nodes)},
                            torch::TensorOptions().dtype(torch::kLong))),
        time_encoder_(time_encoder) {
    register_buffer("memory_", memory_);
    register_buffer("last_update_", last_update_);
    register_buffer("assoc_", assoc_);

    // since our identity msg is cat(mem[src], mem[dst], raw_msg, t_enc)
    const auto cell_dim =
        cfg.memory_dim + cfg.memory_dim + cfg.msg_dim + cfg.time_dim;
    gru_ =
        register_module("gru_", torch::nn::GRUCell(cell_dim, cfg.memory_dim));

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

  auto forward(const torch::Tensor& n_id)
      -> std::tuple<torch::Tensor, torch::Tensor> {
    return is_training() ? get_updated_memory(n_id)
                         : std::make_pair(memory_.index_select(0, n_id),
                                          last_update_.index_select(0, n_id));
  }

  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& t, const torch::Tensor& raw_msg)
      -> void {
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
    const auto msg =
        memory_.new_empty({0, static_cast<std::int64_t>(msg_dim_)});
    const auto empty_entry = std::make_tuple(i, i, i, msg);

    std::fill(src_store_.begin(), src_store_.end(), empty_entry);
    std::fill(dst_store_.begin(), dst_store_.end(), empty_entry);
  }

  auto update_memory(const torch::Tensor& n_id) -> void {
    auto [memory_nid, last_update_nid] = get_updated_memory(n_id);
    memory_.index_put_({n_id}, memory_nid);
    last_update_.index_put_({n_id}, last_update_nid);
  }

  auto get_updated_memory(const torch::Tensor& n_id)
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

  auto update_msg_store(const torch::Tensor& src, const torch::Tensor& dst,
                        const torch::Tensor& t, const torch::Tensor& raw_msg,
                        bool is_src_store) -> void {
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

  auto compute_msg(const torch::Tensor& n_id, bool is_src_store)
      -> const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
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

  auto last_aggr(const torch::Tensor& msg, const torch::Tensor& index,
                 const torch::Tensor& t, int dim_size) -> const torch::Tensor {
    auto out = msg.new_zeros({dim_size, msg.size(-1)});

    // Number of messages is t.numel();
    if (t.numel()) {
      const auto argmax = scatter_argmax(t, index, dim_size);
      const auto mask = argmax < msg.size(0);  // Items with at least one entry
      const auto latest_msgs = msg.index_select(0, argmax.index({mask}));
      out.index_put_({mask}, latest_msgs);
    }

    return out;
  }

  std::size_t msg_dim_{};
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
  TGNImpl(const TGNConfig& cfg, const std::shared_ptr<TGStore>& store)
      : cfg_(cfg),
        store_(std::move(store)),
        nbr_loader_(cfg.num_nbrs, cfg.num_nodes),
        assoc_(torch::full({static_cast<std::int64_t>(cfg.num_nodes)}, -1,
                           torch::TensorOptions().dtype(torch::kLong))) {
    time_encoder_ =
        register_module("time_encoder_", TimeEncoder(cfg_.time_dim));
    memory_ = register_module("memory_", TGNMemory(cfg_, time_encoder_));
    conv_ = register_module(
        "conv_", TransformerConv(cfg_.memory_dim, cfg_.embedding_dim / 2,
                                 cfg_.msg_dim + cfg_.time_dim, cfg_.num_heads,
                                 cfg_.dropout));
  }

  auto reset_state() -> void {
    memory_->reset_state();
    nbr_loader_.reset_state();
  }

  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& t, const torch::Tensor& msg) -> void {
    memory_->update_state(src, dst, t, msg);
    nbr_loader_.insert(src, dst);
  }

  auto detach_memory() -> void { memory_->detach(); }

  template <typename... Ts>
  auto forward(const Ts&... inputs) {
    if constexpr (sizeof...(inputs) == 0) {
      throw std::invalid_argument(
          "TGN::forward requires at least one input ID tensor.");
    }
    std::vector<torch::Tensor> input_list = {inputs...};
    const auto all_global_ids = torch::cat(input_list).view({-1});
    const auto [unique_global_ids, _] = at::_unique(all_global_ids);

    auto compute_embeddings = [&](const torch::Tensor& u_ids) -> torch::Tensor {
      const auto [n_id, edge_index, e_id] = nbr_loader_(u_ids);
      const auto [x, last_update] = memory_(n_id);

      assoc_.index_put_({n_id}, torch::arange(n_id.size(0), assoc_.options()));

      if (edge_index.size(1)) {
        const auto t = store_->fetch_t(e_id);
        const auto msg = store_->fetch_msg(e_id);
        const auto rel_t = last_update.index_select(0, edge_index[0]) - t;
        const auto rel_t_z = time_encoder_->forward(rel_t);
        const auto edge_attr = torch::cat({rel_t_z, msg}, -1);
        return conv_(x, edge_index, edge_attr);
      }
      return x;
    };

    const torch::Tensor z = compute_embeddings(unique_global_ids);

    auto map_to_local = [&](const torch::Tensor& global_n_id) {
      const auto local_indices = assoc_.index({global_n_id});
      return z.index_select(0, local_indices);
    };

    return std::make_tuple(map_to_local(inputs)...);
  }

 private:
  const TGNConfig cfg_{};
  std::shared_ptr<TGStore> store_{};

  TimeEncoder time_encoder_{nullptr};
  TransformerConv conv_{nullptr};
  TGNMemory memory_{nullptr};
  LastNeighborLoader nbr_loader_;

  torch::Tensor assoc_;
};
TORCH_MODULE(TGN);

}  // namespace tgn
