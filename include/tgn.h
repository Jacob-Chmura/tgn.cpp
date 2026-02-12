#include <ATen/ops/rand.h>
#include <torch/nn/module.h>
#include <torch/nn/options/linear.h>
#include <torch/optim/adam.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <shared_mutex>

namespace tgn {

constexpr std::size_t MEMORY_DIM = 100;
constexpr std::size_t TIME_DIM = 100;
constexpr std::size_t EMBEDDING_DIM = 100;

constexpr std::size_t NUM_NODES = 1000;
constexpr std::size_t NUM_NBRS = 10;

const double lr = 1e-3;

struct LinkPredictorImpl : torch::nn::Module {
  LinkPredictorImpl(std::size_t in_channels) {
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
  TimeEncoderImpl(std::size_t out_channels) {
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
    time_encoder = register_module("time_encoder", TimeEncoder(TIME_DIM));

    const auto edge_dim = msg_dim + TIME_DIM;
    conv = register_module(
        "conv", TransformerConv(in_channels, out_channels / 2, edge_dim,
                                2 /*heads*/, 0.1 /* dropout*/));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor last_update,
                        torch::Tensor edge_index, torch::Tensor t,
                        torch::Tensor msg) {
    // rel_t = last_update[edge_index[0]] - t
    // rel_t_enc = self.time_enc(rel_t.to(x.dtype))
    // edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
    // return self.conv(x, edge_index, edge_attr)
    return x;
  }

  TimeEncoder time_encoder{nullptr};
  TransformerConv conv{nullptr};
};
TORCH_MODULE(GraphAttentionEmbedding);

struct LastNeighborLoader {
  LastNeighborLoader() : size(NUM_NBRS), num_nodes(NUM_NODES) {}

  std::size_t size{};
  std::size_t num_nodes{};
};

struct TGNImpl : torch::nn::Module {
  TGNImpl() {
    encoder = register_module(
        "encoder",
        GraphAttentionEmbedding(MEMORY_DIM, EMBEDDING_DIM, 10 /*msg_dim */));
    decoder = register_module("decoder", LinkPredictor(EMBEDDING_DIM));
  }

  auto reset_state() -> void {
    // memory.reset_state();
    // nbr_loader.reset_state();
  }

  auto update_state() -> void {
    // memory.update_state(batch.src, batch.dst, batch.t, batch.msg);
    // nbr_loader.insert(batch.src, batch.dst);
  }

  auto detach_memory() -> void {
    // memory.detach();
  }

  GraphAttentionEmbedding encoder{nullptr};
  LinkPredictor decoder{nullptr};
  LastNeighborLoader nbr_loader{};
};
TORCH_MODULE(TGN);

auto train(TGN tgn, torch::optim::Adam& opt) -> float {
  // Helper vector to map global node indices to local ones.
  // assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
  tgn->train();
  tgn->reset_state();

  // for (auto&batch : *data_loader){
  //}
  opt.zero_grad();

  // n_id, edge_index, e_id = neighbor_loader(batch.n_id)
  // assoc[n_id] = torch.arange(n_id.size(0), device=device)

  // z, last_update = memory(n_id)
  // z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
  // pos_out = decoder(z[assoc[batch.src]], z[assoc[batch.dst]])
  // neg_out = decoder(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
  auto pos_out = torch::rand({10, 1}, torch::requires_grad()).mean();
  auto neg_out = torch::rand({10, 1}, torch::requires_grad()).mean();

  auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
      pos_out, torch::ones_like(pos_out));
  loss += torch::nn::functional::binary_cross_entropy_with_logits(
      neg_out, torch::ones_like(neg_out));

  tgn->update_state();

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
