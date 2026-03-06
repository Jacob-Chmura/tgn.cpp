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
        out_channels_(static_cast<std::int64_t>(out_channels)),
        heads_(static_cast<std::int64_t>(heads)) {
    const auto in_dim = static_cast<std::int64_t>(in_channels);
    const auto out_dim = heads_ * out_channels_;
    w_k_ = register_module("w_k_", torch::nn::Linear(in_dim, out_dim));
    w_q_ = register_module("w_q_", torch::nn::Linear(in_dim, out_dim));
    w_v_ = register_module("w_v_", torch::nn::Linear(in_dim, out_dim));
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
    auto get_us = [](auto start) {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - start)
          .count();
    };

    // Cold Start short-circuit
    if (edge_index.size(1) == 0) {
      return w_skip_->forward(x);
    }

    const auto B = x.size(0);
    const auto E = edge_index.size(1);
    const auto H = heads_;
    const auto C = out_channels_;
    const auto opts = edge_index.options();

    // 1. Projections (Q, K, V, E)
    auto t_proj = std::chrono::steady_clock::now();
    const auto q = w_q_->forward(x).view({B, H, C});
    const auto k = w_k_->forward(x).view({B, H, C});
    const auto v = w_v_->forward(x).view({B, H, C});
    const auto e = w_e_->forward(edge_feat).view({E, H, C});
    auto d_proj = get_us(t_proj);

    // 2. Attention Score Calculation (Dot products)
    auto t_score = std::chrono::steady_clock::now();
    const auto src = edge_index[0];
    const auto dst = edge_index[1];

    const auto k_src = k.index_select(0, src) + e;
    const auto q_dst = q.index_select(0, dst);
    auto alpha = (q_dst * k_src).sum(-1) / std::sqrt(static_cast<double>(C));
    alpha = alpha.view(-1);
    auto d_score = get_us(t_score);

    // 3. Scatter-Softmax (The first potential bottleneck)
    auto t_soft = std::chrono::steady_clock::now();
    const auto H_offset = torch::arange(H, opts).repeat({E});
    auto scatter_idx = (dst.repeat_interleave(H) * H) + H_offset;

    alpha = scatter_softmax(alpha, scatter_idx, B * H);
    alpha = torch::dropout(alpha, dropout_, is_training());
    auto d_soft = get_us(t_soft);

    // 4. Message Aggregation (The "Huge Flatten" and Scatter-Add)
    auto t_aggr = std::chrono::steady_clock::now();
    auto msgs = (v.index_select(0, src) + e) * alpha.view({E, H, 1});
    msgs = msgs.view(-1);

    const auto C_offset = torch::arange(C, opts).repeat({E * H});
    auto scatter_idx_aggr = (scatter_idx.repeat_interleave(C) * C) + C_offset;

    auto out = scatter_add(msgs, scatter_idx_aggr, B * H * C);
    out = out.view({B, H * C});
    auto d_aggr = get_us(t_aggr);

    // 5. Skip connection
    auto t_skip = std::chrono::steady_clock::now();
    auto final_out = out + w_skip_->forward(x);
    auto d_skip = get_us(t_skip);

    std::cout << std::format(
                     "\r[Conv] Proj: {}us | Score: {}us | Softmax: {}us | "
                     "Aggr: {}us | Skip: {}us\n",
                     d_proj, d_score, d_soft, d_aggr, d_skip)
              << std::flush;

    return final_out;
  }

 private:
  torch::nn::Linear w_k_{nullptr}, w_q_{nullptr}, w_v_{nullptr}, w_e_{nullptr},
      w_skip_{nullptr};
  float dropout_{};
  std::int64_t out_channels_{};
  std::int64_t heads_{};
};
TORCH_MODULE(TransformerConv);

struct TGNMemoryImpl : torch::nn::Module {
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
        time_encoder_(time_encoder) {
    register_buffer("memory_", memory_);
    register_buffer("last_update_", last_update_);
    register_buffer("assoc_", assoc_);

    // since our identity msg is cat(mem[src], mem[dst], raw_msg, t_enc)
    const auto cell_dim =
        cfg.memory_dim + cfg.memory_dim + msg_dim_ + cfg.time_dim;
    gru_ =
        register_module("gru_", torch::nn::GRUCell(cell_dim, cfg.memory_dim));

    src_store_.resize(num_nodes_);
    dst_store_.resize(num_nodes_);
    reset_state();

    const auto bytes =
        memory_.nbytes() + last_update_.nbytes() + assoc_.nbytes();
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
    reset_msg_store();
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
      TGN_LOG_DEBUG(
          "TGNMemory: Switching to Eval. Flushing memory for all {} nodes",
          num_nodes_);
      update_memory(torch::arange(static_cast<std::int64_t>(num_nodes_)));
      reset_msg_store();
    }
    torch::nn::Module::train(mode);
  }

 private:
  auto reset_msg_store() -> void {
    // Message store format: (src, dst, t, msg)
    const auto i = torch::empty(0, torch::TensorOptions().dtype(torch::kLong));
    const auto msg = torch::empty({0, static_cast<std::int64_t>(msg_dim_)});
    const auto empty_entry = std::make_tuple(i, i, i, msg);

    std::ranges::fill(src_store_, empty_entry);
    std::ranges::fill(dst_store_, empty_entry);
  }

  auto update_memory(const torch::Tensor& n_id) -> void {
    auto [memory_nid, last_update_nid] = get_updated_memory(n_id);
    memory_.index_put_({n_id}, memory_nid);
    last_update_.index_put_({n_id}, last_update_nid);
  }

  auto get_updated_memory(const torch::Tensor& n_id)
      -> std::tuple<torch::Tensor, torch::Tensor> {
    auto get_us = [](auto start) {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - start)
          .count();
    };

    // 1. Assoc Update
    auto t_assoc = std::chrono::steady_clock::now();
    assoc_.index_put_({n_id}, torch::arange(n_id.size(0), assoc_.options()));
    auto d_assoc = get_us(t_assoc);

    // 2. Message Computation (Src and Dst)
    auto t_msg = std::chrono::steady_clock::now();
    const auto [msg_s, t_s, src_s] = compute_msg(n_id, true);
    const auto [msg_d, t_d, src_d] = compute_msg(n_id, false);
    auto d_msg = get_us(t_msg);

    // 3. Concatenation
    auto t_cat = std::chrono::steady_clock::now();
    const auto idx = torch::cat({src_s, src_d}, 0);
    const auto msg = torch::cat({msg_s, msg_d}, 0);
    const auto t = torch::cat({t_s, t_d}, 0);
    auto d_cat = get_us(t_cat);

    // 4. Aggregation (last_aggr)
    auto t_aggr = std::chrono::steady_clock::now();
    const auto aggr =
        last_aggr(msg, assoc_.index_select(0, idx), t, n_id.size(0));
    auto d_aggr = get_us(t_aggr);

    // 5. GRU / Memory Update
    auto t_gru = std::chrono::steady_clock::now();
    auto updated_memory = gru_->forward(aggr, memory_.index_select(0, n_id));
    auto d_gru = get_us(t_gru);

    // 6. Last Update Calculation (Scatter Max)
    auto t_last = std::chrono::steady_clock::now();
    auto updated_last_update_full = scatter_max(t, idx, last_update_.size(0));
    auto updated_last_update = updated_last_update_full.index_select(0, n_id);
    auto d_last = get_us(t_last);

    std::cout << std::format(
                     "\r[MemUpd] Assoc: {}us | Msg: {}us | Aggr: {}us | GRU: "
                     "{}us | Scat: {}us\n",
                     d_assoc, d_msg, d_aggr, d_gru, d_last)
              << std::flush;

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
    auto* count_data_ptr = count.data_ptr<std::int64_t>();
    std::vector<std::int64_t> sizes(count_data_ptr,
                                    count_data_ptr + count.numel());

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
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    // Gather stored messages
    std::vector<torch::Tensor> src_list;
    std::vector<torch::Tensor> dst_list;
    std::vector<torch::Tensor> t_list;
    std::vector<torch::Tensor> raw_msg_list;

    const auto& store = is_src_store ? src_store_ : dst_store_;
    for (std::int64_t i = 0; i < n_id.numel(); ++i) {
      auto node = n_id[i].item<std::int64_t>();
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

  static auto last_aggr(const torch::Tensor& msg, const torch::Tensor& index,
                        const torch::Tensor& t, std::int64_t dim_size)
      -> torch::Tensor {
    auto out = torch::zeros({dim_size, msg.size(-1)});

    // Number of messages is t.numel();
    if (t.numel() > 0) {
      const auto argmax = scatter_argmax(t, index, dim_size);
      const auto mask = argmax < msg.size(0);  // Items with at least one entry
      const auto latest_msgs = msg.index_select(0, argmax.index({mask}));
      out.index_put_({mask}, latest_msgs);
    }

    return out;
  }

  std::size_t msg_dim_{};
  std::size_t num_nodes_{};
  torch::Tensor memory_;
  torch::Tensor last_update_;
  torch::Tensor assoc_;

  TimeEncoder time_encoder_{nullptr};
  torch::nn::GRUCell gru_{nullptr};

  std::vector<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      src_store_, dst_store_;
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
  auto get_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  auto t_ids_start = std::chrono::steady_clock::now();
  impl_->memory_->update_state(src, dst, time, msg);
  auto d_ids = get_ms(t_ids_start);
  auto t_nbr_start = std::chrono::steady_clock::now();
  impl_->nbr_loader_.insert(src, dst);
  auto d_nbr = get_ms(t_nbr_start);
  std::cout << std::format("\n\t[Mem Upd: {}us][Nbr Upd: {}us]\n", d_ids, d_nbr)
            << std::flush;
}

auto TGNImpl::forward_internal(const std::vector<torch::Tensor>& input_list)
    -> std::vector<torch::Tensor> {
  auto get_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  };

  auto t_ids_start = std::chrono::steady_clock::now();
  const auto all_global_ids = torch::cat(input_list).view({-1});
  const auto [unique_global_ids, _] = at::_unique(all_global_ids);
  auto d_ids = get_ms(t_ids_start);

  // Load neighbors and fetch memory
  auto t_nbr_start = std::chrono::steady_clock::now();
  const auto [n_id, edge_index, e_id] = impl_->nbr_loader_(unique_global_ids);
  auto d_nbr = get_ms(t_nbr_start);
  auto t_mem_start = std::chrono::steady_clock::now();
  const auto [x, last_update] = impl_->memory_->forward(n_id);
  auto d_mem = get_ms(t_mem_start);

  // Update global-to-local buffer
  auto t_assoc_start = std::chrono::steady_clock::now();
  impl_->assoc_.index_put_(
      {n_id}, torch::arange(n_id.size(0), impl_->assoc_.options()));
  auto d_assoc = get_ms(t_assoc_start);

  // Transformer conv with relative time encoding
  auto t_gather_start = std::chrono::steady_clock::now();
  const auto t_edges = impl_->store_->gather_timestamps(e_id);
  const auto raw_msgs = impl_->store_->gather_msgs(e_id);
  auto d_gather = get_ms(t_gather_start);
  auto t_edge_const_start = std::chrono::steady_clock::now();
  const auto rel_t = last_update.index_select(0, edge_index[0]) - t_edges;
  const auto rel_t_z =
      impl_->time_encoder_->forward(rel_t.to(raw_msgs.dtype()));
  const auto edge_feat = torch::cat({rel_t_z, raw_msgs}, -1);
  auto d_edge_const = get_ms(t_edge_const_start);
  auto t_conv_start = std::chrono::steady_clock::now();
  const auto z = impl_->conv_->forward(x, edge_index, edge_feat);
  auto d_conv = get_ms(t_conv_start);

  // Map computed local embeddings back to global id input_list
  std::vector<torch::Tensor> outputs;
  outputs.reserve(input_list.size());
  for (const auto& inp : input_list) {
    const auto local_indices = impl_->assoc_.index({inp});
    outputs.push_back(z.index_select(0, local_indices));
  }
  std::cout << std::format(
                   "\n\t[IDs: {}us][Nbr: {}us][Mem: {}us][Assoc: {}us][Gather: "
                   "{}us][Edge: "
                   "{}us][Conv: {}us]",
                   d_ids, d_nbr, d_mem, d_assoc, d_gather, d_edge_const, d_conv)
            << std::flush;

  return outputs;
}

}  // namespace tgn
