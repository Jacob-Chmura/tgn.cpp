#include "tgn.h"

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "logging.h"
#include "sampler.h"
#include "scatter_ops.h"

namespace tgn {
namespace detail {
struct TimeEncoderImpl : torch::nn::Module {
  explicit TimeEncoderImpl(std::size_t out_channels) {
    lin_ = register_module("lin_", torch::nn::Linear(1, out_channels));
    TGN_LOG_INFO("TimeEncoder: Initialized (time_embedding_dim={})",
                 out_channels);
  }

  auto forward(const torch::Tensor& t) -> torch::Tensor {
    return lin_->forward(t.view({-1, 1})).cos();
  }

 private:
  torch::nn::Linear lin_{nullptr};
};
TORCH_MODULE(TimeEncoder);

struct TransformerConvImpl : torch::nn::Module {
  TransformerConvImpl(std::size_t in_channels, std::size_t out_channels,
                      std::size_t edge_dim, std::size_t heads,
                      float dropout = 0.0)
      : dropout_(dropout),
        H_(static_cast<std::int64_t>(heads)),
        C_(static_cast<std::int64_t>(out_channels)) {
    H_offsets_ = register_buffer("H_offsets", torch::arange(H_, torch::kLong));
    C_offsets_ = register_buffer("C_offsets", torch::arange(C_, torch::kLong));

    const auto in_dim = static_cast<std::int64_t>(in_channels);
    const auto out_dim = H_ * C_;
    w_kqv_ = register_module("w_qkv_", torch::nn::Linear(in_dim, 3 * out_dim));
    w_skip_ = register_module("w_skip_", torch::nn::Linear(in_dim, out_dim));
    w_e_ = register_module(
        "w_e_",
        torch::nn::Linear(torch::nn::LinearOptions(
                              static_cast<std::int64_t>(edge_dim), out_dim)
                              .bias(false)));
    TGN_LOG_INFO(
        "TransformerConv: Initialized (in_channels={}, out_channels={}, "
        "heads={}, edge_dim={}, dropout={:.2f})",
        in_channels, out_channels, heads, edge_dim, dropout);
  }

  auto forward(const torch::Tensor& x, const torch::Tensor& edge_index,
               const torch::Tensor& edge_feat) -> torch::Tensor {
    // Cold Start short-circuit (no edges sampled for this batch)
    if (edge_index.size(1) == 0) {
      return w_skip_->forward(x);
    }

    // TODO(kuba): implement 2d scatter ops to avoid these huge flatten ops
    const auto B = x.size(0);
    const auto E = edge_index.size(1);

    // Projections
    const auto qkv = w_kqv_->forward(x).view({B, 3, H_, C_});
    const auto e = w_e_->forward(edge_feat).view({E, H_, C_});

    // Attention scores
    const auto src = edge_index[0];  // src is the sender
    const auto dst = edge_index[1];  // dst is the receiver

    const auto k_src = qkv.select(1, 0).index_select(0, src) + e;
    const auto q_dst = qkv.select(1, 1).index_select(0, dst);
    auto alpha = (q_dst * k_src).sum(-1).div(std::sqrt(C_));
    alpha = alpha.view(-1);  // flatten for 2-d scatter [E * H]

    // Scatter-softmax attention
    const auto H_offset = H_offsets_.expand({E, H_}).reshape(-1);
    auto scatter_idx =
        (dst.unsqueeze(-1).expand({E, H_}).reshape(-1) * H_) + H_offset;

    alpha = scatter_softmax(alpha, scatter_idx, B * H_);
    alpha = torch::dropout(alpha, dropout_, is_training());

    // Scatter-add message aggregation
    auto msgs =
        (qkv.select(1, 2).index_select(0, src) + e) * alpha.view({E, H_, 1});
    msgs = msgs.view(-1);  // flatten for 3-d scatter [E * H * C]

    const auto C_offset = C_offsets_.expand({E * H_, C_}).reshape(-1);
    scatter_idx =
        (scatter_idx.unsqueeze(-1).expand({E * H_, C_}).reshape(-1) * C_) +
        C_offset;

    auto out = scatter_add(msgs, scatter_idx, B * H_ * C_);
    out = out.view({B, H_ * C_});

    return out + w_skip_->forward(x);
  }

 private:
  torch::Tensor H_offsets_, C_offsets_;
  torch::nn::Linear w_kqv_{nullptr}, w_e_{nullptr}, w_skip_{nullptr};
  float dropout_{};
  std::int64_t H_{}, C_{};
};
TORCH_MODULE(TransformerConv);

struct TGNMemoryImpl : torch::nn::Module {
  struct MsgStore {
    torch::Tensor src_, dst_, time_, msg_;

    MsgStore(std::int64_t num_nodes, std::int64_t msg_dim) {
      src_ = torch::zeros({num_nodes}, torch::kLong);
      dst_ = torch::zeros({num_nodes}, torch::kLong);
      time_ = torch::zeros({num_nodes}, torch::kLong);
      msg_ = torch::zeros({num_nodes, msg_dim}, torch::kFloat);
    }

    auto reset() -> void {
      src_.zero_();
      dst_.zero_();
      time_.zero_();
      msg_.zero_();
    }

    auto update(const torch::Tensor& src, const torch::Tensor& dst,
                const torch::Tensor& time, const torch::Tensor msg) -> void {
      // Find the index of the last (max time) interaction for each source node
      auto argmax = scatter_argmax(time, src, src_.size(0));

      // mask out nodes that didn't appear in this batch
      auto mask = argmax < src.size(0);
      auto active_node_ids = torch::nonzero(mask).view(-1);
      auto batch_indices = argmax.index({mask});

      src_.index_put_({active_node_ids}, src.index_select(0, batch_indices));
      dst_.index_put_({active_node_ids}, dst.index_select(0, batch_indices));
      time_.index_put_({active_node_ids}, time.index_select(0, batch_indices));
      msg_.index_put_({active_node_ids}, msg.index_select(0, batch_indices));
    }
  };

  explicit TGNMemoryImpl(const TGNConfig& cfg, const TimeEncoder& time_encoder,
                         std::int64_t msg_dim, std::int64_t num_nodes)
      : msg_dim_(msg_dim),
        num_nodes_(num_nodes),
        memory_(torch::empty(
            {num_nodes, static_cast<std::int64_t>(cfg.memory_dim)})),
        last_update_(torch::empty({num_nodes},
                                  torch::TensorOptions().dtype(torch::kLong))),
        assoc_(torch::empty({num_nodes},
                            torch::TensorOptions().dtype(torch::kLong))),
        time_encoder_(time_encoder),
        src_store_(num_nodes, msg_dim),
        dst_store_(num_nodes, msg_dim) {
    register_buffer("memory_", memory_);
    register_buffer("last_update_", last_update_);
    register_buffer("assoc_", assoc_);

    // since our identity msg is cat(mem[src], mem[dst], raw_msg, t_enc)
    const auto cell_dim =
        cfg.memory_dim + cfg.memory_dim + msg_dim_ + cfg.time_dim;
    gru_ =
        register_module("gru_", torch::nn::GRUCell(cell_dim, cfg.memory_dim));

    reset_state();

    auto get_store_bytes = [](const MsgStore& s) {
      return s.src_.nbytes() + s.dst_.nbytes() + s.time_.nbytes() +
             s.msg_.nbytes();
    };

    const auto bytes = memory_.nbytes() + last_update_.nbytes() +
                       assoc_.nbytes() + get_store_bytes(src_store_) +
                       get_store_bytes(dst_store_);
    TGN_LOG_INFO(
        "TGNMemory: ~{:.2f} MiB allocated ({} nodes, memory_dim: {}, msg_dim: "
        "{}, gru_cell_dim: {})",
        bytes / (1024.0 * 1024.0), num_nodes_, cfg.memory_dim, msg_dim_,
        cell_dim);
  }

  auto reset_state() -> void {
    TGN_LOG_DEBUG("TGNMemory: Resetting state");
    memory_.zero_();
    last_update_.zero_();
    src_store_.reset();
    dst_store_.reset();
  }

  auto detach() -> void { memory_.detach_(); }

  auto forward(const torch::Tensor& n_id)
      -> std::tuple<torch::Tensor, torch::Tensor> {
    return is_training() ? get_updated_memory(n_id)
                         : std::make_tuple(memory_.index_select(0, n_id),
                                           last_update_.index_select(0, n_id));
  }

  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& t, const torch::Tensor& raw_msg)
      -> void {
    const auto [n_id, _] = at::_unique(torch::cat({src, dst}));

    if (is_training()) {
      update_memory(n_id);
      src_store_.update(src, dst, t, raw_msg);
      dst_store_.update(dst, src, t, raw_msg);
    } else {
      src_store_.update(src, dst, t, raw_msg);
      dst_store_.update(dst, src, t, raw_msg);
      update_memory(n_id);
    }
  }

  auto train(bool mode = true) -> void override {
    if (is_training() && !mode) {
      // Flush message store in case we just entered eval mode.
      TGN_LOG_DEBUG(
          "TGNMemory: Switching to Eval. Flushing memory for all {} nodes",
          num_nodes_);
      update_memory(torch::arange(static_cast<std::int64_t>(num_nodes_)));
      src_store_.reset();
      dst_store_.reset();
    }
    torch::nn::Module::train(mode);
  }

 private:
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

    auto last_aggr = [&](const torch::Tensor& _msg, const torch::Tensor& _index,
                         const torch::Tensor& _t,
                         std::int64_t _dim_size) -> torch::Tensor {
      auto out = torch::zeros({_dim_size, _msg.size(-1)});

      // Number of messages is t.numel();
      if (_t.numel() > 0) {
        const auto argmax = scatter_argmax(_t, _index, _dim_size);
        const auto mask =
            argmax < _msg.size(0);  // Items with at least one entry
        const auto latest_msgs = _msg.index_select(0, argmax.index({mask}));
        out.index_put_({mask}, latest_msgs);
      }

      return out;
    };

    const auto aggr =
        last_aggr(msg, assoc_.index_select(0, idx), t, n_id.size(0));

    // Get local copy of updated memory, and then last_update.
    auto updated_memory = gru_->forward(aggr, memory_.index_select(0, n_id));
    auto updated_last_update = scatter_max(t, idx, last_update_.size(0));

    updated_last_update = updated_last_update.index_select(0, n_id);
    return {updated_memory, updated_last_update};
  }

  auto compute_msg(const torch::Tensor& n_id, bool is_src_store)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    const auto& store = is_src_store ? src_store_ : dst_store_;

    // Find which nodes in the current batch have messages in oru store
    const auto mask = store.time_.index_select(0, n_id) > 0;
    const auto active_n_id = n_id.index({mask});

    // Gather message
    const auto src = store.src_.index_select(0, active_n_id);
    const auto dst = store.dst_.index_select(0, active_n_id);
    const auto t = store.time_.index_select(0, active_n_id);
    const auto raw_msg = store.msg_.index_select(0, active_n_id);

    // Compute msg components
    const auto rel_t = t - last_update_.index_select(0, src);
    const auto rel_t_z = time_encoder_->forward(rel_t.to(raw_msg.dtype()));
    const auto mem_src = memory_.index_select(0, src);
    const auto mem_dst = memory_.index_select(0, dst);

    // Final message (identity aggr)
    const auto msg = torch::cat({mem_src, mem_dst, raw_msg, rel_t_z}, 1);

    return std::make_tuple(msg, t, src);
  }

  std::size_t msg_dim_{};
  std::size_t num_nodes_{};
  torch::Tensor memory_;
  torch::Tensor last_update_;
  torch::Tensor assoc_;

  TimeEncoder time_encoder_{nullptr};
  torch::nn::GRUCell gru_{nullptr};

  MsgStore src_store_, dst_store_;
};
TORCH_MODULE(TGNMemory);
}  // namespace detail

struct TGNImpl::Impl {
  Impl(const TGNConfig& cfg, const std::shared_ptr<TGStore>& store)
      : cfg_(cfg),
        store_(store),
        nbr_loader_(cfg.num_nbrs, store->node_count()),
        assoc_(torch::full({static_cast<std::int64_t>(store->node_count())}, -1,
                           torch::dtype(torch::kLong))) {
    time_encoder_ = detail::TimeEncoder(cfg.time_dim);
    conv_ = detail::TransformerConv(cfg.memory_dim, cfg.embedding_dim / 2,
                                    store->msg_dim() + cfg.time_dim,
                                    cfg.num_heads, cfg.dropout);
    memory_ = detail::TGNMemory(cfg, time_encoder_, store->msg_dim(),
                                store->node_count());
  }

  const TGNConfig cfg_;
  std::shared_ptr<TGStore> store_;
  detail::TimeEncoder time_encoder_{nullptr};
  detail::TransformerConv conv_{nullptr};
  detail::TGNMemory memory_{nullptr};
  LastNeighborLoader nbr_loader_;
  torch::Tensor assoc_;
};

TGNImpl::TGNImpl(const TGNConfig& cfg, const std::shared_ptr<TGStore>& store)
    : impl_(std::make_unique<Impl>(cfg, store)) {
  register_module("time_encoder", impl_->time_encoder_);
  register_module("memory", impl_->memory_);
  register_module("conv", impl_->conv_);

  impl_->assoc_ = register_buffer("assoc", impl_->assoc_);
}

TGNImpl::~TGNImpl() = default;

auto TGNImpl::detach_memory() -> void { impl_->memory_->detach(); }

auto TGNImpl::reset_state() -> void {
  impl_->memory_->reset_state();
  impl_->nbr_loader_.reset_state();
}

auto TGNImpl::update_state(const torch::Tensor& src, const torch::Tensor& dst,
                           const torch::Tensor& time, const torch::Tensor& msg)
    -> void {
  impl_->memory_->update_state(src, dst, time, msg);
  impl_->nbr_loader_.insert(src, dst);
}

auto TGNImpl::forward_internal(const std::vector<torch::Tensor>& input_list)
    -> std::vector<torch::Tensor> {
  const auto all_global_ids = torch::cat(input_list).view({-1});
  const auto [unique_global_ids, _] = at::_unique(all_global_ids);

  // Load neighbors and fetch memory
  const auto [n_id, edge_index, e_id] = impl_->nbr_loader_(unique_global_ids);
  const auto [x, last_update] = impl_->memory_->forward(n_id);

  // Update global-to-local buffer
  impl_->assoc_.index_put_(
      {n_id}, torch::arange(n_id.size(0), impl_->assoc_.options()));

  // Transformer conv with relative time encoding
  const auto t_edges = impl_->store_->gather_timestamps(e_id);
  const auto raw_msgs = impl_->store_->gather_msgs(e_id);
  const auto rel_t = last_update.index_select(0, edge_index[0]) - t_edges;
  const auto rel_t_z =
      impl_->time_encoder_->forward(rel_t.to(raw_msgs.dtype()));
  const auto edge_feat = torch::cat({rel_t_z, raw_msgs}, -1);
  const auto z = impl_->conv_->forward(x, edge_index, edge_feat);

  // Map computed local embeddings back to global id input_list
  std::vector<torch::Tensor> outputs;
  outputs.reserve(input_list.size());
  for (const auto& inp : input_list) {
    const auto local_indices = impl_->assoc_.index({inp});
    outputs.push_back(z.index_select(0, local_indices));
  }

  return outputs;
}

}  // namespace tgn
