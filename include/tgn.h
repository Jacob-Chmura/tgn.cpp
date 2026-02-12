#include <ATen/ops/rand.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/adam.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <shared_mutex>
#include <tuple>
#include <utility>

namespace tgn {

constexpr std::size_t MEMORY_DIM = 100;
constexpr std::size_t TIME_DIM = 100;
constexpr std::size_t EMBEDDING_DIM = 100;

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
  explcit TimeEncoderImpl(std::size_t out_channels) {
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
        "lin_edge", torch::nn::Linear(torch::nn::LinearOptions(
                                          in_channels, heads * out_channels)
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

struct GraphAttentionEmbeddingImpl : torch::nn::Module {
  GraphAttentionEmbeddingImpl(std::size_t in_channels, std::size_t out_channels,
                              std::size_t msg_dim) {
    // TODO(kuba): this should be shared with TGNMemory
    time_encoder = register_module("time_encoder", TimeEncoder(TIME_DIM));

    const auto edge_dim = msg_dim + TIME_DIM;
    conv = register_module(
        "conv", TransformerConv(in_channels, out_channels / 2, edge_dim,
                                2 /*heads*/, 0.1 /* dropout*/));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor last_update,
                        torch::Tensor edge_index, torch::Tensor t,
                        torch::Tensor msg) {
    auto rel_t = last_update[edge_index[0]] - t;
    auto rel_t_enc = time_encoder->forward(rel_t);
    auto edge_attr = torch::cat({rel_t_enc, msg}, -1);
    return conv(x, edge_index, edge_attr);
  }

  TimeEncoder time_encoder{nullptr};
  TransformerConv conv{nullptr};
};
TORCH_MODULE(GraphAttentionEmbedding);

struct LastNeighborLoader {
  LastNeighborLoader() : size(NUM_NBRS), num_nodes(NUM_NODES) {
    // self.nbrs = torch.empty((num_nodes, size), dtype=torch.long,
    // device=device) self.e_id = torch.empty((num_nodes, size),
    // dtype=torch.long, device=device) self._assoc = torch.empty(num_nodes,
    // dtype=torch.long, device=device)

    // self.reset_state()
  }

  auto operator()(torch::Tensor n_id) -> void {
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
  TGNMemoryImpl() : num_nodes(NUM_NODES) {
    // super().__init__()

    // self.num_nodes = num_nodes
    // self.raw_msg_dim = raw_msg_dim

    // self.aggr_module = aggr_module
    // self.time_enc = TimeEncoder(time_dim)
    // self.gru = GRUCell(msg_module.out_channels, memory_dim)

    // self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
    // self.register_buffer("last_update", torch.empty(num_nodes,
    // dtype=torch.long)) self.register_buffer("_assoc", torch.empty(num_nodes,
    // dtype=torch.long))

    // self.msg_s_store = {}
    // self.msg_d_store = {}

    // self.reset_parameters()
  }

  auto reset_state() -> void {
    // zeros(self.memory)
    // zeros(self.last_update)
    // self._reset_message_store()
  }

  auto detach() -> void {
    // self.memory.detach_();
  }

  auto forward(torch::Tensor n_id) -> std::tuple<torch::Tensor, torch::Tensor> {
    // return is_training() ? _get_updated_memory() :  // memory[n_id],
    // last_update[n_id];
    return {torch::rand({1, 1}), torch::rand({1, 1})};
  }

  auto update_state(torch::Tensor src, torch::Tensor dst, torch::Tensor t,
                    torch::Tensor raw_msg) -> void {
    // n_id = torch.cat([src, dst]).unique()

    // if self.training:
    //     self._update_memory(n_id)
    //     self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
    //     self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    // else:
    //     self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
    //     self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    //     self._update_memory(n_id)
  }

  /*

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst), then (dst -> src).
        msg_s, t_s, src_s, _= self._compute_msg(n_id, self.msg_s_store)
        msg_d, t_d, src_d, _= self._compute_msg(n_id, self.msg_d_store)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory, and then last_update.
        memory = self.gru(aggr, self.memory[n_id])
        last_update = scatter(t, idx, 0, dim_size=self.last_update.size(0),
reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, msg_store
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0).to(self.device)
        dst = torch.cat(dst, dim=0).to(self.device)
        t = torch.cat(t, dim=0).to(self.device)
        # Filter out empty tensors to avoid `invalid configuration argument`.
        # TODO Investigate why this is needed.
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = torch.cat(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, src, dst

    def train(self, mode: bool = True):
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes,
device=self.memory.device)) self._reset_message_store() super().train(mode)


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
};
TORCH_MODULE(TGNMemory);

struct TGNImpl : torch::nn::Module {
  TGNImpl() {
    encoder = register_module(
        "encoder",
        GraphAttentionEmbedding(MEMORY_DIM, EMBEDDING_DIM, 10 /*msg_dim */));
    memory = register_module("memory", TGNMemory());
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

  torch::Tensor forward(torch::Tensor n_id) {
    // n_id, edge_index, e_id = nbr_loader(n_id)
    // assoc[n_id] = torch.arange(n_id.size(0), device=device)
    // z, last_update = memory(n_id)
    // z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
    return torch::rand({10, 1}, torch::requires_grad()).mean();
  }

  GraphAttentionEmbedding encoder{nullptr};
  TGNMemory memory{nullptr};
  LastNeighborLoader nbr_loader{};
};
TORCH_MODULE(TGN);

auto train(TGN tgn, torch::optim::Adam& opt) -> float {
  // Helper vector to map global node indices to local ones.
  // assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
  tgn->train();
  tgn->reset_state();

  opt.zero_grad();

  auto src = torch::rand({1, 1});
  auto dst = torch::rand({1, 1});
  auto neg_dst = torch::rand({1, 1});
  auto t = torch::rand({1, 1});
  auto msg = torch::rand({1, 1});
  auto n_id = torch::rand({1, 1});

  auto z = tgn->forward(n_id);

  // pos_out = decoder(z[assoc[batch.src]], z[assoc[batch.dst]]);
  // neg_out = decoder(z[assoc[batch.src]], z[assoc[batch.neg_dst]]);
  auto pos_out = torch::rand({10, 1}, torch::requires_grad()).mean();
  auto neg_out = torch::rand({10, 1}, torch::requires_grad()).mean();

  auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
      pos_out, torch::ones_like(pos_out));
  loss += torch::nn::functional::binary_cross_entropy_with_logits(
      neg_out, torch::ones_like(neg_out));

  tgn->update_state(src, dst, t, msg);

  loss.backward();
  opt.step();

  tgn->detach_memory();

  return loss.item<float>();
}

auto hello_torch() -> void {
  TGN tgn;
  torch::optim::Adam opt(tgn->parameters(), torch::optim::AdamOptions(lr));

  for (std::size_t epoch = 1; epoch <= 2; ++epoch) {
    auto loss = train(tgn, opt);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
  }
}

}  // namespace tgn
