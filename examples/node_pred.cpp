#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "lib.h"
#include "util.h"

constexpr std::size_t num_epochs = 10;
constexpr std::size_t batch_size = 200;
constexpr double learning_rate = 1e-4;
constexpr std::string dataset = "tgbn-genre";

namespace {

struct NodePredictorImpl : torch::nn::Module {
  explicit NodePredictorImpl(std::size_t in_dim, std::size_t hidden_dim = 64) {
    model_->push_back(torch::nn::Linear(in_dim, hidden_dim));
    model_->push_back(torch::nn::ReLU());
    model_->push_back(torch::nn::Linear(hidden_dim, 1));

    register_module("model", model_);
  }

  auto forward(const torch::Tensor& z_node) -> torch::Tensor {
    return model_->forward(z_node);
  }

 private:
  torch::nn::Sequential model_{nullptr};
};
TORCH_MODULE(NodePredictor);

auto compute_ndcg(const torch::Tensor& y_pred, const torch::Tensor& y_true,
                  std::int64_t k = 10) -> float {
  k = std::min(k, y_pred.size(-1));

  const auto [_, top_indices] = y_pred.topk(k, -1);
  const auto y_true_at_pred_topk = y_true.gather(-1, top_indices);

  const auto ranks = torch::arange(1, k + 1).to(torch::kFloat32);
  const auto discounts = torch::log2(ranks + 1.0);
  const auto dcg = (y_true_at_pred_topk / discounts).sum(-1);

  const auto [ideal_labels, _] = y_true.topk(k, -1);
  const auto idcg = (ideal_labels / discounts).sum(-1);

  const auto ndcg = dcg / (idcg + 1e-8);
  return ndcg.mean().item<float>();
}

auto train(tgn::TGN& encoder, NodePredictor& decoder, torch::optim::Adam& opt,
           const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};
  auto e_id = store->train_split().start();

  for (; e_id < store->train_split().end(); e_id += batch_size) {
    opt.zero_grad();

    const auto batch = store->get_batch(e_id, batch_size);
    const auto [z_src] = encoder->forward(batch.src);

    const auto y_true = 0;  // batch.node_y
    const auto y_pred = decoder->forward(z_src);

    auto loss = torch::nn::functional::cross_entropy(y_pred, y_true);
    loss.backward();
    opt.step();
    total_loss += loss.item<float>();

    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);
    encoder->detach_memory();

    util::progress_bar(e_id - store->train_split().start(),
                       store->train_split().size(), "Train", start_time);
  }
  std::cout << std::endl;
  return total_loss / static_cast<float>(store->train_split().size());
}

auto eval(tgn::TGN& encoder, NodePredictor& decoder,
          const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();

  torch::NoGradGuard no_grad;
  encoder->eval();
  decoder->eval();

  std::vector<float> perf_list;
  auto e_id = store->val_split().start();

  for (; e_id < store->val_split().end(); e_id += batch_size) {
    const auto batch = store->get_batch(e_id, batch_size);
    const auto [z_src] = encoder->forward(batch.src);

    const auto y_true = 0;  // batch.node_y
    const auto y_pred = decoder->forward(z_src);

    perf_list.push_back(compute_ndcg(y_pred, y_true));
    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);

    util::progress_bar(e_id - store->val_split().start(),
                       store->val_split().size(), "Valid", start_time);
  }
  std::cout << std::endl;
  return perf_list.empty()
             ? 0.0
             : std::accumulate(perf_list.begin(), perf_list.end(), 0.0) /
                   perf_list.size();
}

}  // namespace

auto main() -> int {
  const auto cfg = tgn::TGNConfig{};
  const auto opts = util::load_csv("data/" + dataset + ".csv");
  const auto store = tgn::make_store(opts);

  tgn::TGN encoder(cfg, store);
  NodePredictor decoder{cfg.embedding_dim};

  auto params = encoder->parameters();
  auto dec_params = decoder->parameters();
  params.insert(params.end(), dec_params.begin(), dec_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(learning_rate));

  for (std::size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    auto loss = train(encoder, decoder, opt, store);
    auto ndcg = eval(encoder, decoder, store);
    std::cout << "Epoch " << epoch << " Loss: " << loss << " NDCG: " << ndcg
              << std::endl;
  }
}
