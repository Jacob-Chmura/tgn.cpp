#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>

#include "lib.h"

// Learning params
constexpr std::size_t NUM_EPOCHS = 10;
constexpr std::size_t BATCH_SIZE = 5;
constexpr double LEARNING_RATE = 1e-3;

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
    const auto z = torch::relu(w_src->forward(z_src) + w_dst->forward(z_dst));
    return w_final->forward(z);
  }

 private:
  torch::nn::Linear w_src{nullptr}, w_dst{nullptr}, w_final{nullptr};
};
TORCH_MODULE(LinkPredictor);

auto train(tgn::TGN& encoder, LinkPredictor& decoder, torch::optim::Adam& opt,
           const std::shared_ptr<tgn::TGStore>& store) -> float {
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};
  std::size_t e_id = 0;

  for (; e_id < store->num_edges(); e_id += BATCH_SIZE) {
    opt.zero_grad();

    const auto batch = store->get_batch(e_id, BATCH_SIZE);
    const auto [z_src, z_dst, z_neg] =
        encoder->forward(batch.src, batch.dst, batch.neg_dst);

    const auto pos_out = decoder->forward(z_src, z_dst);
    const auto neg_out = decoder->forward(z_src, z_neg);

    auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    pos_out, torch::ones_like(pos_out)) +
                torch::nn::functional::binary_cross_entropy_with_logits(
                    neg_out, torch::zeros_like(neg_out));

    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);
    loss.backward();
    opt.step();
    encoder->detach_memory();

    total_loss += loss.item<float>();
  }
  return total_loss / static_cast<float>(e_id);
}
}  // namespace

auto main() -> int {
  const auto cfg = tgn::TGNConfig{};
  const auto store_opts =
      tgn::InMemoryTGStoreOptions{.src = torch::randint(0, 1000, {100}),
                                  .dst = torch::randint(0, 1000, {100}),
                                  .t = torch::arange(100).to(torch::kFloat),
                                  .msg = torch::rand({100, 7}),
                                  .neg_dst = torch::randint(0, 1000, {100})};

  const auto store = tgn::make_store(store_opts);

  tgn::TGN encoder(cfg, store);
  LinkPredictor decoder{cfg.embedding_dim};

  auto params = encoder->parameters();
  auto dec_params = decoder->parameters();
  params.insert(params.end(), dec_params.begin(), dec_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(LEARNING_RATE));

  for (std::size_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
    auto loss = train(encoder, decoder, opt, store);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
  }
}
