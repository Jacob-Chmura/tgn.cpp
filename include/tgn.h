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
    lin_key = register_module(
        "lin_key", torch::nn::Linear(in_channels, heads * out_channels));
    lin_query = register_module(
        "lin_query", torch::nn::Linear(in_channels, heads * out_channels));
    lin_value = register_module(
        "lin_value", torch::nn::Linear(in_channels, heads * out_channels));
    lin_skip = register_module(
        "lin_skip", torch::nn::Linear(in_channels, heads * out_channels));
    lin_edge = register_module(
        "lin_edge", torch::nn::Linear(
                        torch::nn::LinearOptions(edge_dim, heads * out_channels)
                            .bias(false)));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index,
                        torch::Tensor edge_attr) {
    const auto num_nodes = x.size(0);
    auto j = edge_index[0];  // j is the sender
    auto i = edge_index[1];  // i is the receiver

    auto q = lin_query->forward(x).view({-1, heads, out_channels});
    auto k = lin_key->forward(x).view({-1, heads, out_channels});
    auto v = lin_value->forward(x).view({-1, heads, out_channels});
    auto e = lin_edge->forward(edge_attr).view({-1, heads, out_channels});

    // Message Function
    auto key_j = k.index_select(0, j) + e;
    auto query_i = q.index_select(0, i);
    auto alpha = (query_i * key_j).sum(-1) /
                 std::sqrt(static_cast<double>(out_channels));

    // Scatter softmax
    // a = torch_geometric.softmax(a, index=i, size_i=w_q.size(0))
    alpha = torch::dropout(alpha, dropout, is_training());

    auto out = (v.index_select(0, j) + e) * alpha.view({-1, heads, 1});
    // Scatter add
    // out = aggr_module(out, i, dim=0, dim_size=w_q.size(0));

    out = out.view({num_nodes, heads * out_channels});
    return out + lin_skip->forward(x);
  }

  torch::nn::Linear lin_key{nullptr}, lin_query{nullptr}, lin_value{nullptr},
      lin_edge{nullptr}, lin_skip{nullptr};
  float dropout{};
  std::size_t out_channels{};
  std::size_t heads{};
};
TORCH_MODULE(TransformerConv);

struct LastNeighborLoader {
  LastNeighborLoader() : size(NUM_NBRS), num_nodes(NUM_NODES) {
    // self.nbrs = torch.empty((num_nodes, size), dtype=torch.long)
    // self.e_id = torch.empty((num_nodes, size), dtype=torch.long)
    // self._assoc = torch.empty(num_nodes, dtype=torch.long)
    reset_state();
  }

  auto operator()(torch::Tensor global_n_id)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    // nbrs = self.nbrs[n_id]
    // nodes = n_id.view(-1, 1).repeat(1, self.size)
    // e_id = self.e_id[n_id]

    // # Filter invalid nbrs (identified by `e_id < 0`).
    // mask = e_id >= 0
    // nbrs, nodes, e_id = nbrs[mask], nodes[mask], e_id[mask]

    // # Relabel node indices.
    // n_id = torch.cat([n_id, nbrs]).unique()
    // self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
    // nbrs, nodes = self._assoc[nbrs], self._assoc[nodes]

    // return n_id, torch.stack([nbrs, nodes]), e_id
    auto [n_id, _] = at::_unique(global_n_id);

    auto indices = torch::arange(n_id.size(0), n_id.options());
    std::vector<torch::Tensor> indices_list = {indices, indices};
    torch::Tensor edge_index = torch::stack(indices_list, 0);

    torch::Tensor e_id =
        torch::zeros({n_id.size(0)}, n_id.options().dtype(torch::kLong));

    return {n_id, edge_index, e_id};
  }

  auto insert(torch::Tensor src, torch::Tensor dst) -> void {
    // # Collect central nodes, their nbrs and the current event ids.
    // nbrs = torch.cat([src, dst], dim=0)
    // nodes = torch.cat([dst, src], dim=0)
    // e_id = torch.arange(
    //     self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
    //).repeat(2)
    // self.cur_e_id += src.numel()

    // # Convert newly encountered interaction ids so that they point to
    // # locations of a "dense" format of shape [num_nodes, size].
    // nodes, perm = nodes.sort()
    // nbrs, e_id = nbrs[perm], e_id[perm]

    // n_id = nodes.unique()
    // self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

    // dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
    // dense_id += self._assoc[nodes].mul_(self.size)

    // dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
    // dense_e_id[dense_id] = e_id
    // dense_e_id = dense_e_id.view(-1, self.size)

    // dense_nbrs = e_id.new_empty(n_id.numel() * self.size)
    // dense_nbrs[dense_id] = nbrs
    // dense_nbrs = dense_nbrs.view(-1, self.size)

    // # Collect new and old interactions...
    // e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
    // nbrs = torch.cat([self.nbrs[n_id, : self.size], dense_nbrs], dim=-1)

    // # And sort them based on `e_id`.
    // e_id, perm = e_id.topk(self.size, dim=-1)
    // self.e_id[n_id] = e_id
    // self.nbrs[n_id] = torch.gather(nbrs, 1, perm)
  }

  auto reset_state() -> void {
    // self.cur_e_id = 0
    // self.e_id.fill_(-1)
  }

  std::size_t size{};
  std::size_t num_nodes{};
};

struct TGNMemoryImpl : torch::nn::Module {
  TGNMemoryImpl()
      : num_nodes(NUM_NODES),
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

    // self.aggr_module = aggr_module
    // self.time_enc = TimeEncoder(time_dim)
    // self.msg_s_store = {}
    // self.msg_d_store = {}

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
    // self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

    // # Compute messages (src -> dst), then (dst -> src).
    // msg_s, t_s, src_s, _= self._compute_msg(n_id, self.msg_s_store)
    // msg_d, t_d, src_d, _= self._compute_msg(n_id, self.msg_d_store)

    // # Aggregate messages.
    // idx = torch.cat([src_s, src_d], dim=0)
    // msg = torch.cat([msg_s, msg_d], dim=0)
    // t = torch.cat([t_s, t_d], dim=0)
    // aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

    // # Get local copy of updated memory, and then last_update.
    // memory = self.gru(aggr, self.memory[n_id])
    // last_update = scatter(t, idx, 0, dim_size=self.last_update.size(0),
    // reduce="//max")[n_id]

    // return memory, last_update
    //
    auto x = torch::zeros({n_id.size(0), MEMORY_DIM});
    auto last_update =
        torch::zeros({n_id.size(0)}, n_id.options().dtype(torch::kLong));
    return {x, last_update};
  }

  auto _update_msg_store(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                         torch::Tensor raw_msg, bool is_src_store) -> void {
    //     n_id, perm = src.sort()
    //     n_id, count = n_id.unique_consecutive(return_counts=True)
    //     for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
    //         msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
  }

  auto _compute_msg() -> void {
    // def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType):
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
  }

  auto train() -> void {
    // def train(self, mode: bool = True):
    //    if self.training and not mode:
    //        # Flush message store in case we just entered eval mode.
    //        self._update_memory(torch.arange(self.num_nodes,
    //        device=self.memory.device)) self._reset_message_store()
    // super().train(mode)
  }

  /*
class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out

class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
   */

  std::size_t num_nodes{};
  torch::Tensor memory{};
  torch::Tensor last_update{};
  torch::Tensor assoc{};

  torch::nn::GRUCell gru{nullptr};
};
TORCH_MODULE(TGNMemory);

struct TGNImpl : torch::nn::Module {
  TGNImpl() {
    memory = register_module("memory", TGNMemory());
    time_encoder = register_module("time_encoder", TimeEncoder(TIME_DIM));
    conv =
        register_module("conv", TransformerConv(MEMORY_DIM, EMBEDDING_DIM / 2,
                                                MSG_DIM + TIME_DIM, 2 /*heads*/,
                                                0.1 /* dropout*/));
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

    // Problem: global ref to data
    // auto t = data_t.index_select(0, e_id);
    // auto msg = data_msg.index_select(0, e_id);
    auto t = torch::rand({n_id.size(0)});
    auto msg = torch::rand({n_id.size(0), MSG_DIM});

    auto rel_t = last_update.index_select(0, edge_index[0]) - t;
    auto rel_t_z = time_encoder->forward(rel_t);
    auto edge_attr = torch::cat({rel_t_z, msg}, -1);

    z_cache = conv(x, edge_index, edge_attr);
    n_id_cache = n_id;

    return z_cache;
  }

  auto get_embeddings(torch::Tensor global_n_id) -> torch::Tensor {
    // Build the local mapping (assoc) on the fly for the cached nodes
    auto assoc = torch::full({static_cast<std::uint64_t>(NUM_NODES)}, -1,
                             torch::TensorOptions().dtype(torch::kLong));
    assoc.index_put_({n_id_cache},
                     torch::arange(n_id_cache.size(0), assoc.options()));

    auto local_indices = assoc.index({global_n_id});
    return z_cache.index_select(0, local_indices);
  }

  TimeEncoder time_encoder{nullptr};
  TransformerConv conv{nullptr};
  TGNMemory memory{nullptr};
  LastNeighborLoader nbr_loader{};

  torch::Tensor z_cache;
  torch::Tensor n_id_cache;
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
