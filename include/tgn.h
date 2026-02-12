#include <ATen/ops/_unique.h>
#include <ATen/ops/rand.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/rnn.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/adam.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace tgn {

constexpr std::size_t MEMORY_DIM = 100;
constexpr std::size_t TIME_DIM = 100;
constexpr std::size_t MSG_DIM = 7;
constexpr std::size_t EMBEDDING_DIM = 100;
constexpr std::size_t BATCH_SIZE = 5;
constexpr std::size_t NUM_NODES = 1000;
constexpr std::size_t NUM_NBRS = 10;
constexpr std::size_t NUM_HEADS = 2;
constexpr float DROPOUT = 0.1;
const double lr = 1e-3;

struct LinkPredictorImpl : torch::nn::Module {
  explicit LinkPredictorImpl(std::size_t in_channels) {
    lin_src =
        register_module("lin_src", torch::nn::Linear(in_channels, in_channels));
    lin_dst =
        register_module("lin_dst", torch::nn::Linear(in_channels, in_channels));
    lin_final = register_module("lin_final", torch::nn::Linear(in_channels, 1));
  }

  torch::Tensor forward(torch::Tensor z_src, torch::Tensor z_dst) {
    auto z = torch::relu(lin_src->forward(z_src) + lin_dst->forward(z_dst));
    return lin_final->forward(z);
  }

  torch::nn::Linear lin_src{nullptr}, lin_dst{nullptr}, lin_final{nullptr};
};
TORCH_MODULE(LinkPredictor);

struct TimeEncoderImpl : torch::nn::Module {
  explicit TimeEncoderImpl(std::size_t out_channels) {
    lin = register_module("lin", torch::nn::Linear(1, out_channels));
  }

  torch::Tensor forward(torch::Tensor t) {
    return lin->forward(t.view({-1, 1})).cos();
  }

  torch::nn::Linear lin{nullptr};
};
TORCH_MODULE(TimeEncoder);

struct TransformerConvImpl : torch::nn::Module {
  TransformerConvImpl(std::size_t in_channels, std::size_t out_channels,
                      std::size_t edge_dim, std::size_t heads,
                      float dropout = 0.0)
      : dropout(dropout), out_channels(out_channels), heads(heads) {
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
    auto j = edge_index[0];  // j is the sender
    auto i = edge_index[1];  // i is the receiver

    auto q = w_q->forward(x).view({-1, heads, out_channels});
    auto k = w_k->forward(x).view({-1, heads, out_channels});
    auto v = w_v->forward(x).view({-1, heads, out_channels});
    auto e = w_e->forward(edge_attr).view({-1, heads, out_channels});

    // Message Function
    auto key_j = k.index_select(0, j) + e;
    auto query_i = q.index_select(0, i);
    auto alpha = (query_i * key_j).sum(-1) /
                 std::sqrt(static_cast<double>(out_channels));

    // TODO(kuba): Scatter softmax
    // a = torch_geometric.softmax(a, index=i, size_i=w_q.size(0))
    alpha = torch::dropout(alpha, dropout, is_training());

    auto out = (v.index_select(0, j) + e) * alpha.view({-1, heads, 1});
    // TODO(kuba): Scatter add
    // out = aggr_module(out, i, dim=0, dim_size=w_q.size(0));

    out = out.view({num_nodes, heads * out_channels});
    return out + w_skip->forward(x);
  }

  torch::nn::Linear w_k{nullptr}, w_q{nullptr}, w_v{nullptr}, w_e{nullptr},
      w_skip{nullptr};
  float dropout{};
  std::size_t out_channels{};
  std::size_t heads{};
};
TORCH_MODULE(TransformerConv);

struct LastNeighborLoader {
  LastNeighborLoader()
      : size(NUM_NBRS),
        num_nodes(NUM_NODES),
        buffer_nbrs(torch::empty({NUM_NODES, NUM_NBRS},
                                 torch::TensorOptions().dtype(torch::kLong))),
        buffer_e_id(torch::empty({NUM_NODES, NUM_NBRS},
                                 torch::TensorOptions().dtype(torch::kLong))),
        assoc(torch::empty({NUM_NODES},
                           torch::TensorOptions().dtype(torch::kLong))) {
    reset_state();
  }

  auto operator()(torch::Tensor global_n_id)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    // Shape: [batch_size, sampler_size]
    auto nbrs = buffer_nbrs.index_select(0, global_n_id);
    auto e_id = buffer_e_id.index_select(0, global_n_id);

    // Shape: [batch_size, sampler_size]
    auto nodes = global_n_id.view({-1, 1}).expand_as(nbrs);

    // Filter invalid neighbors (e_id < 0)
    auto mask = e_id >= 0;
    auto filtered_nbrs = nbrs.index({mask});
    auto filtered_e_id = e_id.index({mask});
    auto filtered_nodes = nodes.index({mask});

    // Relabel node indices and combine nodes with sampled neighbors
    auto [unique_n_id, _] =
        at::_unique(torch::cat({global_n_id, filtered_nbrs}));
    assoc.index_put_({unique_n_id},
                     torch::arange(unique_n_id.size(0), unique_n_id.options()));

    if (filtered_nbrs.numel() == 0) {
      auto empty_edge_index =
          torch::empty({2, 0}, global_n_id.options().dtype(torch::kLong));
      return std::make_tuple(unique_n_id, empty_edge_index, filtered_e_id);
    }

    // Map global IDs to local IDs [0, len(n_id) - 1]
    auto local_nbrs = assoc.index_select(0, filtered_nbrs);
    auto local_nodes = assoc.index_select(0, filtered_nodes);
    auto edge_index = torch::stack({local_nbrs, local_nodes}, 0);

    return std::make_tuple(unique_n_id, edge_index, filtered_e_id);
  }

  auto insert(torch::Tensor src, torch::Tensor dst) -> void {
    // Collect central nodes, their nbrs and the current event ids.
    auto nbrs = torch::cat({src, dst}, 0);
    auto nodes = torch::cat({dst, src}, 0);

    // Create edge IDs for this batch and repeat for bi-directional edges
    auto batch_size = src.size(0);
    auto e_id_range =
        torch::arange(cur_e_id, cur_e_id + batch_size, src.options());
    auto e_id = e_id_range.repeat({2});
    cur_e_id += batch_size;

    // Sort interactions by node ID to simplify batch processing
    auto [sort_out, perm] = nodes.sort();
    nodes = sort_out;
    nbrs = nbrs.index_select(0, perm);
    e_id = e_id.index_select(0, perm);

    // Find unique nodes and map to local range [0, num_unique - 1]
    auto [unique_out, _] = at::_unique(nodes);
    auto n_id = unique_out;
    assoc.index_put_({n_id}, torch::arange(n_id.size(0), n_id.options()));

    // Create "dense" temporary representation
    // dense_id determines the column in the [num_unique, size] window
    auto dense_id = torch::arange(nodes.size(0), nodes.options()) % size;
    dense_id += assoc.index_select(0, nodes).mul_(size);

    auto total_temp_slots = n_id.size(0) * size;

    auto dense_e_id = torch::full({total_temp_slots}, -1, e_id.options());
    dense_e_id.index_put_({dense_id}, e_id);
    dense_e_id = dense_e_id.view({-1, size});

    auto dense_nbrs = torch::empty({total_temp_slots}, nbrs.options());
    dense_nbrs.index_put_({dense_id}, nbrs);
    dense_nbrs = dense_nbrs.view({-1, size});

    // Merge new interactions with existing ones in the global buffers
    // Fetch old data for the relevant nodes: shape [num_unique, size]
    auto old_e_id = buffer_e_id.index_select(0, n_id);
    auto old_nbrs = buffer_nbrs.index_select(0, n_id);

    // Concatenate old and new: shape [num_unique, size * 2]
    auto merged_e_id = torch::cat({old_e_id, dense_e_id}, -1);
    auto merged_nbrs = torch::cat({old_nbrs, dense_nbrs}, -1);

    // Keep only the 'size' most recent interactions (highest e_id)
    auto topk_out = merged_e_id.topk(size, -1);
    auto new_e_id = std::get<0>(topk_out);
    auto topk_perm = std::get<1>(topk_out);

    // Use gather to pick the corresponding neighbors
    auto new_nbrs = torch::gather(merged_nbrs, 1, topk_perm);

    // Write back to global buffers
    buffer_e_id.index_put_({n_id}, new_e_id);
    buffer_nbrs.index_put_({n_id}, new_nbrs);
  }

  auto reset_state() -> void {
    cur_e_id = 0;
    buffer_e_id.fill_(-1);
  }

  std::size_t size{};
  std::size_t num_nodes{};
  std::size_t cur_e_id{};

  torch::Tensor buffer_nbrs;
  torch::Tensor buffer_e_id;
  torch::Tensor assoc;
};

struct TGNMemoryImpl : torch::nn::Module {
  explicit TGNMemoryImpl(TimeEncoder time_encoder)
      : time_encoder(time_encoder),
        num_nodes(NUM_NODES),
        memory(torch::empty({NUM_NODES, MEMORY_DIM})),
        last_update(torch::empty({NUM_NODES},
                                 torch::TensorOptions().dtype(torch::kLong))),
        assoc(torch::empty({NUM_NODES},
                           torch::TensorOptions().dtype(torch::kLong))) {
    register_buffer("memory", memory);
    register_buffer("last_update", last_update);
    register_buffer("assoc", assoc);

    // since our identity msg is cat(mem[src], mem[dst], raw_msg, t_enc)
    constexpr auto cell_dim = MEMORY_DIM + MEMORY_DIM + MSG_DIM + TIME_DIM;
    gru = register_module("gru", torch::nn::GRUCell(cell_dim, MEMORY_DIM));

    reset_state();
  }

  auto reset_state() -> void {
    memory.zero_();
    last_update.zero_();
    _reset_message_store();
  }

  auto detach() -> void { memory.detach_(); }

  auto forward(torch::Tensor n_id) -> std::tuple<torch::Tensor, torch::Tensor> {
    return is_training() ? _get_updated_memory(n_id)
                         : std::make_pair(memory.index_select(0, n_id),
                                          last_update.index_select(0, n_id));
  }

  auto update_state(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                    torch::Tensor raw_msg) -> void {
    auto [n_id, _] = at::_unique(torch::cat({src, dst}));

    if (is_training()) {
      _update_memory(n_id);
      _update_msg_store(src, dst, t, raw_msg, true);
      _update_msg_store(dst, src, t, raw_msg, false);
    } else {
      _update_msg_store(src, dst, t, raw_msg, true);
      _update_msg_store(dst, src, t, raw_msg, false);
      _update_memory(n_id);
    }
  }

  auto _reset_message_store() -> void {
    // TODO(kuba)
    // i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
    // msg = self.memory.new_empty((0, self.raw_msg_dim),
    // device=self.device) # Message store format: (src, dst, t, msg)
    // self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
    // self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
  }

  auto _update_memory(torch::Tensor n_id) -> void {
    auto [memory_nid, last_update_nid] = _get_updated_memory(n_id);
    memory.index_put_({n_id}, memory_nid);
    last_update.index_put_({n_id}, last_update_nid);
  }

  auto _get_updated_memory(torch::Tensor n_id)
      -> std::tuple<torch::Tensor, torch::Tensor> {
    assoc.index_put_({n_id}, torch::arange(n_id.size(0)));

    // Compute messages (src -> dst), then (dst -> src).
    auto [msg_s, t_s, src_s] = _compute_msg(n_id, true);
    auto [msg_d, t_d, src_d] = _compute_msg(n_id, false);

    // Aggregate messages.
    auto idx = torch::cat({src_s, src_d}, 0);
    auto msg = torch::cat({msg_s, msg_d}, 0);
    auto t = torch::cat({t_s, t_d}, 0);
    auto aggr = last_aggr(msg, assoc.index_select(0, idx), t, n_id.size(0));

    // Get local copy of updated memory, and then last_update.
    auto updated_memory = gru->forward(aggr, memory.index_select(0, n_id));
    // TODO(kuba)
    // last_update = scatter(t, idx, 0, dim_size=last_update.size(0),
    // reduce="max")[n_id]
    auto last_update =
        torch::zeros({n_id.size(0)}, n_id.options().dtype(torch::kLong));
    return {updated_memory, last_update};
  }

  auto _update_msg_store(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                         torch::Tensor raw_msg, bool is_src_store) -> void {
    // TODO(kuba)
    //     n_id, perm = src.sort()
    //     n_id, count = n_id.unique_consecutive(return_counts=True)
    //     for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
    //         msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
  }

  auto _compute_msg(torch::Tensor n_id, bool is_src_store)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    // TODO(kuba)
    //     data = [msg_store[i] for i in n_id.tolist()]
    //     src, dst, t, raw_msg = list(zip(*data))
    //     src = torch.cat(src, dim=0).to(self.device)
    //     dst = torch.cat(dst, dim=0).to(self.device)
    //     t = torch.cat(t, dim=0).to(self.device)
    //     # Filter out empty tensors to avoid `invalid configuration argument`.
    //     # TODO Investigate why this is needed.
    //     raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i ==
    //     0] raw_msg = torch.cat(raw_msg, dim=0).to(self.device) t_rel = t -
    //     self.last_update[src] t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

    //    msg = torch.cat(self.memory[src], self.memory[dst], raw_msg, t_enc)
    //    return msg, t, src, dst
    return std::make_tuple(torch::rand({n_id.size(0), MEMORY_DIM + MEMORY_DIM +
                                                          MSG_DIM + TIME_DIM}),
                           torch::rand({n_id.size(0)}),
                           torch::randint(0, 1, n_id.size(0)));
  }

  auto last_aggr(torch::Tensor msg, torch::Tensor index, torch::Tensor t,
                 int dim_size) -> torch::Tensor {
    // TODO(kuba)
    // argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
    // out = msg.new_zeros((dim_size, msg.size(-1)))
    // mask = argmax < msg.size(0)  # Filter items with at least one entry.
    // out[mask] = msg[argmax[mask]]
    // return out
    return torch::zeros({dim_size, msg.size(-1)}, msg.options());
  }

  auto train(bool mode = true) -> void override {
    if (is_training() && !mode) {
      // Flush message store in case we just entered eval mode.
      _update_memory(torch::arange(num_nodes));
      _reset_message_store();
    }
    torch::nn::Module::train(mode);
  }

  std::size_t num_nodes{};
  torch::Tensor memory{};
  torch::Tensor last_update{};
  torch::Tensor assoc{};

  TimeEncoder time_encoder{nullptr};
  torch::nn::GRUCell gru{nullptr};
};
TORCH_MODULE(TGNMemory);

struct TGNImpl : torch::nn::Module {
  TGNImpl()
      : assoc(torch::full({NUM_NODES}, -1,
                          torch::TensorOptions().dtype(torch::kLong))) {
    time_encoder = register_module("time_encoder", TimeEncoder(TIME_DIM));
    memory = register_module("memory", TGNMemory(time_encoder));
    conv = register_module(
        "conv", TransformerConv(MEMORY_DIM, EMBEDDING_DIM / 2,
                                MSG_DIM + TIME_DIM, NUM_HEADS, DROPOUT));
  }

  auto reset_state() -> void {
    memory->reset_state();
    nbr_loader.reset_state();
  }

  auto update_state(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                    torch::Tensor msg) -> void {
    memory->update_state(src, dst, t, msg);
    nbr_loader.insert(src, dst);
  }

  auto detach_memory() -> void { memory->detach(); }

  torch::Tensor forward(torch::Tensor global_n_id) {
    auto [n_id, edge_index, e_id] = nbr_loader(global_n_id);
    auto [x, last_update] = memory(n_id);

    assoc.index_put_({n_id}, torch::arange(n_id.size(0), assoc.options()));

    if (e_id.numel() > 0) {
      // TODO(kuba): global ref to data
      auto t = torch::rand({n_id.size(0)});             // data.t[e_id]
      auto msg = torch::rand({n_id.size(0), MSG_DIM});  // data.msg[e_id]

      auto rel_t = last_update.index_select(0, edge_index[0]) - t;
      auto rel_t_z = time_encoder->forward(rel_t);
      auto edge_attr = torch::cat({rel_t_z, msg}, -1);

      z_cache = conv(x, edge_index, edge_attr);
    } else {
      z_cache = x;
    }

    return z_cache;
  }

  auto get_embeddings(torch::Tensor global_n_id) -> torch::Tensor {
    auto local_indices = assoc.index({global_n_id});
    return z_cache.index_select(0, local_indices);
  }

  TimeEncoder time_encoder{nullptr};
  TransformerConv conv{nullptr};
  TGNMemory memory{nullptr};
  LastNeighborLoader nbr_loader{};

  torch::Tensor z_cache;
  torch::Tensor assoc;
};
TORCH_MODULE(TGN);

struct Batch {
  torch::Tensor src, dst, neg_dst, t, msg;
};

auto get_batch() -> Batch {
  return Batch{
      .src = torch::randint(0, NUM_NODES, {BATCH_SIZE}),
      .dst = torch::randint(0, NUM_NODES, {BATCH_SIZE}),
      .neg_dst = torch::randint(0, NUM_NODES, {BATCH_SIZE}),
      .t = torch::zeros({BATCH_SIZE}),
      .msg = torch::zeros({BATCH_SIZE, MSG_DIM}),
  };
}

auto train(TGN tgn, LinkPredictor decoder, torch::optim::Adam& opt) -> float {
  tgn->train();
  decoder->train();
  tgn->reset_state();

  float loss_{0};
  {
    auto batch = get_batch();
    opt.zero_grad();

    auto [n_id, _] =
        at::_unique(torch::cat({batch.src, batch.dst, batch.neg_dst}));
    tgn->forward(n_id);

    auto z_src = tgn->get_embeddings(batch.src);
    auto z_dst = tgn->get_embeddings(batch.dst);
    auto z_neg = tgn->get_embeddings(batch.neg_dst);

    auto pos_out = decoder->forward(z_src, z_dst);
    auto neg_out = decoder->forward(z_src, z_neg);

    auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    pos_out, torch::ones_like(pos_out)) +
                torch::nn::functional::binary_cross_entropy_with_logits(
                    neg_out, torch::zeros_like(neg_out));
    std::cout << "Loss: " << loss << std::endl;

    tgn->update_state(batch.src, batch.dst, batch.t, batch.msg);
    loss.backward();
    opt.step();
    tgn->detach_memory();

    loss_ = loss.item<float>();
  }

  return loss_;
}

auto hello_torch() -> void {
  TGN encoder;
  LinkPredictor decoder{EMBEDDING_DIM};

  std::vector<torch::Tensor> params = encoder->parameters();
  auto decoder_params = decoder->parameters();
  params.insert(params.end(), decoder_params.begin(), decoder_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(lr));

  for (std::size_t epoch = 1; epoch <= 5; ++epoch) {
    auto loss = train(encoder, decoder, opt);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
  }
}

}  // namespace tgn
