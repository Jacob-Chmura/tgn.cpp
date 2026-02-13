#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include "tgn.h"

// Learning params
constexpr std::size_t NUM_EPOCHS = 10;
constexpr std::size_t BATCH_SIZE = 5;
const double lr = 1e-3;

namespace {

struct LinkPredictorImpl : torch::nn::Module {
  explicit LinkPredictorImpl(std::size_t in_channels) {
    w_src =
        register_module("w_src", torch::nn::Linear(in_channels, in_channels));
    w_dst =
        register_module("w_dst", torch::nn::Linear(in_channels, in_channels));
    w_final = register_module("w_final", torch::nn::Linear(in_channels, 1));
  }

  torch::Tensor forward(torch::Tensor z_src, torch::Tensor z_dst) {
    auto z = torch::relu(w_src->forward(z_src) + w_dst->forward(z_dst));
    return w_final->forward(z);
  }

  torch::nn::Linear w_src{nullptr}, w_dst{nullptr}, w_final{nullptr};
};
TORCH_MODULE(LinkPredictor);

struct Batch {
  torch::Tensor src, dst, neg_dst, t, msg;
};

auto get_batch() -> Batch {
  return Batch{
      .src = torch::randint(0, tgn::NUM_NODES, {BATCH_SIZE}),
      .dst = torch::randint(0, tgn::NUM_NODES, {BATCH_SIZE}),
      .neg_dst = torch::randint(0, tgn::NUM_NODES, {BATCH_SIZE}),
      .t = torch::zeros({BATCH_SIZE}),
      .msg = torch::zeros({BATCH_SIZE, tgn::MSG_DIM}),
  };
}

auto train(tgn::TGN tgn, LinkPredictor decoder, torch::optim::Adam& opt)
    -> float {
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

    tgn->update_state(batch.src, batch.dst, batch.t, batch.msg);
    loss.backward();
    opt.step();
    tgn->detach_memory();

    loss_ = loss.item<float>();
  }

  return loss_;
}

}  // namespace
auto main() -> int {
  tgn::TGN encoder;
  LinkPredictor decoder{tgn::EMBEDDING_DIM};

  std::vector<torch::Tensor> params = encoder->parameters();
  auto decoder_params = decoder->parameters();
  params.insert(params.end(), decoder_params.begin(), decoder_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(lr));

  for (std::size_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
    auto loss = train(encoder, decoder, opt);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
  }
}
