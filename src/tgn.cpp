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
#include "tguf.h"

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
        C_(static_cast<std::int64_t>(out_channels)),
        O_(static_cast<std::int64_t>(heads * out_channels)) {
    const auto in_dim = static_cast<std::int64_t>(in_channels);
    w_kqv_ = register_module("w_qkv_", torch::nn::Linear(in_dim, 3 * O_));
    w_skip_ = register_module("w_skip_", torch::nn::Linear(in_dim, O_));
    w_e_ = register_module(
        "w_e_", torch::nn::Linear(torch::nn::LinearOptions(
                                      static_cast<std::int64_t>(edge_dim), O_)
                                      .bias(false)));
    TGN_LOG_INFO(
        "TransformerConv: Initialized (in_channels={}, out_channels={}, "
        "heads={}, edge_dim={}, dropout={:.2f})",
        in_channels, out_channels, heads, edge_dim, dropout);
  }

  auto forward(const torch::Tensor& x, const torch::Tensor& edge_index,
               const torch::Tensor& edge_feat) -> torch::Tensor {
    if (edge_index.size(1) == 0) {
      return w_skip_->forward(x);
    }

    const auto B = x.size(0);
    const auto E = edge_index.size(1);
    const auto src = edge_index[0];  // src is the sender
    const auto dst = edge_index[1];  // dst is the receiver

    // Projections
    const auto qkv = w_kqv_->forward(x);
    const auto q = qkv.narrow(1, 0, O_).view({B, H_, C_});
    const auto k = qkv.narrow(1, O_, O_).view({B, H_, C_});
    const auto v = qkv.narrow(1, 2 * O_, O_).view({B, H_, C_});
    const auto e = w_e_->forward(edge_feat).view({E, H_, C_});

    // Attention scores
    const auto k_src = k.index_select(0, src) + e;
    const auto q_dst = q.index_select(0, dst);
    auto alpha = (q_dst * k_src).sum(-1).div(std::sqrt(C_));

    // Scatter-softmax attention
    alpha = scatter_softmax(alpha, dst, B);
    alpha = torch::dropout(alpha, dropout_, is_training());

    // Scatter-add message aggregation
    const auto msgs = (v.index_select(0, src) + e) * alpha.view({E, H_, 1});
    const auto out = scatter_add(msgs.reshape({E, O_}), dst, B);

    return out + w_skip_->forward(x);
  }

 private:
  torch::nn::Linear w_kqv_{nullptr}, w_e_{nullptr}, w_skip_{nullptr};
  float dropout_{};
  std::int64_t H_{}, C_{}, O_{};
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
    // Check which store has the most recent interaction for each node
    const auto t_s = src_store_.time_.index_select(0, n_id);
    const auto t_d = dst_store_.time_.index_select(0, n_id);
    const auto src_is_newer = t_s >= t_d;
    const auto has_msg = (t_s > 0) | (t_d > 0);

    // Gather winning metadata
    const auto t = torch::where(src_is_newer, t_s, t_d);
    const auto src =
        torch::where(src_is_newer, src_store_.src_.index_select(0, n_id),
                     dst_store_.src_.index_select(0, n_id));
    const auto dst =
        torch::where(src_is_newer, src_store_.dst_.index_select(0, n_id),
                     dst_store_.dst_.index_select(0, n_id));
    const auto raw_msg = torch::where(src_is_newer.unsqueeze(1),
                                      src_store_.msg_.index_select(0, n_id),
                                      dst_store_.msg_.index_select(0, n_id));

    // Compute msg
    const auto last_upd_src = last_update_.index_select(0, src);
    const auto rel_t = (t - last_upd_src).to(raw_msg.dtype());
    const auto rel_t_z = time_encoder_->forward(rel_t);

    const auto mem_src = memory_.index_select(0, src);
    const auto mem_dst = memory_.index_select(0, dst);

    auto aggr = torch::cat({mem_src, mem_dst, raw_msg, rel_t_z}, 1);
    aggr = torch::where(has_msg.unsqueeze(1), aggr, torch::zeros_like(aggr));

    // Get updated memory, and last_update, if a message actually existed
    const auto last_update = last_update_.index_select(0, n_id);
    const auto memory = memory_.index_select(0, n_id);

    auto updated_last_update = torch::where(has_msg, t, last_update);
    auto updated_memory = gru_->forward(aggr, memory);
    updated_memory = torch::where(has_msg.unsqueeze(1), updated_memory, memory);

    return {updated_memory, updated_last_update};
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
  Impl(const TGNConfig& cfg, const std::shared_ptr<tguf::TGStore>& store)
      : cfg_(cfg),
        store_(store),
        nbr_loader_(cfg.num_nbrs, store->node_count()),
        assoc_(torch::full({static_cast<std::int64_t>(store->node_count())}, -1,
                           torch::dtype(torch::kLong))) {
    time_encoder_ = detail::TimeEncoder(cfg.time_dim);
    conv_ = detail::TransformerConv(
        cfg.memory_dim + store_->node_feat_dim(), cfg.embedding_dim / 2,
        store->msg_dim() + cfg.time_dim, cfg.num_heads, cfg.dropout);
    memory_ = detail::TGNMemory(cfg, time_encoder_, store->msg_dim(),
                                store->node_count());
  }

  const TGNConfig cfg_;
  std::shared_ptr<tguf::TGStore> store_;
  detail::TimeEncoder time_encoder_{nullptr};
  detail::TransformerConv conv_{nullptr};
  detail::TGNMemory memory_{nullptr};
  LastNeighborLoader nbr_loader_;
  torch::Tensor assoc_;
};

TGNImpl::TGNImpl(const TGNConfig& cfg,
                 const std::shared_ptr<tguf::TGStore>& store)
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
  const auto [memory, last_update] = impl_->memory_->forward(n_id);

  // Update global-to-local buffer
  impl_->assoc_.index_put_(
      {n_id}, torch::arange(n_id.size(0), impl_->assoc_.options()));

  // Transformer conv with relative time encoding
  const auto t_edges = impl_->store_->gather_timestamps(e_id);
  const auto rel_t = last_update.index_select(0, edge_index[0]) - t_edges;
  const auto rel_t_z = impl_->time_encoder_->forward(rel_t.to(torch::kFloat32));
  const auto edge_feat =
      impl_->store_->msg_dim() > 0
          ? torch::cat({rel_t_z, impl_->store_->gather_msgs(e_id)}, -1)
          : rel_t_z;
  const auto node_feat =
      impl_->store_->node_feat_dim() > 0
          ? torch::cat({memory, impl_->store_->gather_node_feats(n_id)}, -1)
          : memory;
  const auto z = impl_->conv_->forward(node_feat, edge_index, edge_feat);

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
