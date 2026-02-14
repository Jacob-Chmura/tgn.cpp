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
    msg_dim_ = static_cast<std::int64_t>(MSG_DIM);
  }

  auto size() const -> std::size_t override { return n_events_; }

  [[nodiscard]] auto get_batch(std::size_t start, std::size_t batch_size) const
      -> Batch override {
    const auto end = std::min(start + batch_size, n_events_);
    const auto current_batch_size = static_cast<std::int64_t>(end - start);
    return Batch{
        .src = torch::randint(0, tgn::NUM_NODES, {current_batch_size}),
        .dst = torch::randint(0, tgn::NUM_NODES, {current_batch_size}),
        .t = torch::arange(static_cast<std::int64_t>(start),
                           static_cast<std::int64_t>(end), torch::kLong),
        .msg = torch::zeros({current_batch_size, tgn::MSG_DIM}),
        .neg_dst = torch::randint(0, tgn::NUM_NODES, {current_batch_size}),
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
    const auto num_nodes = x.size(0);
    const auto num_edges = edge_index.size(1);
    const auto q = w_q->forward(x).view({num_nodes, heads_, out_channels_});
    const auto k = w_k->forward(x).view({num_nodes, heads_, out_channels_});
    const auto v = w_v->forward(x).view({num_nodes, heads_, out_channels_});
    const auto e =
        w_e->forward(edge_attr).view({num_edges, heads_, out_channels_});

    // Attention
    const auto j = edge_index[0];  // j is the sender
    const auto i = edge_index[1];  // i is the receiver

    const auto k_j = k.index_select(0, j) + e;
    const auto q_i = q.index_select(0, i);

    auto a =
        (q_i * k_j).sum(-1) / std::sqrt(static_cast<double>(out_channels_));
    auto a_flat = a.view(-1);
    auto i_flat = i.repeat_interleave(heads_);
    auto head_offset = torch::arange(heads_, i.options()).repeat({num_edges});
    i_flat += (head_offset * num_nodes);

    a_flat = scatter_softmax(a_flat, i_flat, /* dim_size */ num_nodes * heads_);
    a_flat = torch::dropout(a_flat, dropout_, is_training());

    a = a_flat.view({num_edges, heads_});

    // TODO(kuba): implement 2d scatter ops to avoid these huge flatten ops
    auto msgs = (v.index_select(0, j) + e) * a.view({num_edges, heads_, 1});
    auto m_flat = msgs.view(-1);
    auto i_super_flat = i_flat.repeat_interleave(out_channels_);
    auto chan_offset =
        torch::arange(out_channels_, i.options()).repeat({num_edges * heads_});
    i_super_flat += (chan_offset * (num_nodes * heads_));

    auto out_flat =
        scatter_add(m_flat, i_super_flat,
                    /*dim_size*/ num_nodes * heads_ * out_channels_);

    auto out = out_flat.view({num_nodes, heads_ * out_channels_});
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
  explicit TGNMemoryImpl(TimeEncoder time_encoder)
      : num_nodes_(NUM_NODES),
        memory_(torch::empty({NUM_NODES, MEMORY_DIM})),
        last_update_(torch::empty({NUM_NODES},
                                  torch::TensorOptions().dtype(torch::kLong))),
        assoc_(torch::empty({NUM_NODES},
                            torch::TensorOptions().dtype(torch::kLong))),
        time_encoder_(time_encoder) {
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
    const auto msg = memory_.new_empty({0, MSG_DIM});
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
    const auto tt = torch::cat({t_s, t_d}, 0);

    const auto aggr =
        last_aggr(msg, assoc_.index_select(0, idx), tt, n_id.size(0));

    // Get local copy of updated memory, and then last_update.
    auto updated_memory = gru_->forward(aggr, memory_.index_select(0, n_id));
    auto updated_last_update = scatter_max(tt, idx, last_update_.size(0));

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
  TGNImpl(const std::shared_ptr<TGStore>& store)
      : store_(std::move(store)),
        nbr_loader_(NUM_NBRS, NUM_NODES),
        assoc_(torch::full({NUM_NODES}, -1,
                           torch::TensorOptions().dtype(torch::kLong))) {
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

    compute_embeddings(unique_global_ids);

    return std::make_tuple(get_embeddings(inputs)...);
  }

  auto compute_embeddings(const torch::Tensor& unique_global_ids) -> void {
    const auto [n_id, edge_index, e_id] = nbr_loader_(unique_global_ids);
    const auto [x, last_update] = memory_(n_id);
    const std::int64_t num_edges = edge_index.size(1);

    assoc_.index_put_({n_id}, torch::arange(n_id.size(0), assoc_.options()));

    if (num_edges) {
      const auto t = store_->fetch_t(e_id);
      const auto msg = store_->fetch_msg(e_id);
      const auto rel_t = last_update.index_select(0, edge_index[0]) - t;
      const auto rel_t_z = time_encoder_->forward(rel_t);
      const auto edge_attr = torch::cat({rel_t_z, msg}, -1);
      z_cache_ = conv_(x, edge_index, edge_attr);
    } else {
      z_cache_ = x;
    }
  }

  auto get_embeddings(const torch::Tensor& global_n_id) -> const torch::Tensor {
    const auto local_indices = assoc_.index({global_n_id});
    return z_cache_.index_select(0, local_indices);
  }

 private:
  std::shared_ptr<TGStore> store_{};

  TimeEncoder time_encoder_{nullptr};
  TransformerConv conv_{nullptr};
  TGNMemory memory_{nullptr};
  LastNeighborLoader nbr_loader_;

  torch::Tensor z_cache_;
  torch::Tensor assoc_;
};
TORCH_MODULE(TGN);

}  // namespace tgn
