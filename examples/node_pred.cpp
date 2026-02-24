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
constexpr std::size_t node_batch_size = 200;
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

  const auto [e_idx_start, e_idx_end] = store->train_e_idx_range();
  const auto [l_idx_start, l_idx_end] = store->train_label_event_range();
  auto e_idx = e_idx_start;
  auto l_idx = l_idx_start;

  while (e_idx < e_idx_end || l_idx < l_idx_end) {
    // Determine the next e_idx until which we can update model state
    const auto stop_e_idx = (l_idx < l_idx_end)
                                ? store->get_stop_e_idx_for_label_event(l_idx)
                                : e_idx_end;

    // Consume edges until we hit stop_e_id
    if (e_idx < stop_e_idx) {
      const auto step = std::min(batch_size, stop_e_idx - e_idx);
      const auto batch = store->get_batch(e_idx, step);

      encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);
      e_idx += step;
      encoder->detach_memory();
    }
    // We are exactly at a stop_e_id, do Node Property Prediction
    else if (l_idx < l_idx_end) {
      const auto [n_id, y_true] = store->get_label_event(l_idx++);
      const auto num_nodes = n_id.size(0);
      float event_loss{0};

      for (auto i = 0; i < num_nodes; i += node_batch_size) {
        const auto end = std::min(i + node_batch_size, num_nodes);
        const auto n_id_batch = n_id.slice(0, i, end);
        const auto y_true_batch = n_id.slice(0, i, end);

        opt.zero_grad();

        const auto [z] = encoder->forward(n_id);
        const auto y_pred = decoder->forward(z);

        auto loss = torch::nn::functional::cross_entropy(y_pred, y_true);
        loss.backward();
        opt.step();
        event_loss +=
            loss.item<float>() * (static_cast<float>(end - i) / num_nodes);
      }
      total_loss += event_loss;
    }

    util::progress_bar(e_idx - e_idx_start, e_idx_end - e_idx_start, "Train",
                       start_time);
  }

  std::cout << std::endl;
  return total_loss / static_cast<float>(l_idx_end - l_idx_start);
}

auto eval(tgn::TGN& encoder, NodePredictor& decoder,
          const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();

  torch::NoGradGuard no_grad;
  encoder->eval();
  decoder->eval();

  std::vector<float> perf_list;

  const auto [e_idx_start, e_idx_end] = store->val_e_idx_range();
  const auto [l_idx_start, l_idx_end] = store->val_label_event_range();
  auto e_idx = e_idx_start;
  auto l_idx = l_idx_start;

  while (e_idx < e_idx_end || l_idx < l_idx_end) {
    const auto stop_e_idx = (l_idx < l_idx_end)
                                ? store->get_stop_e_idx_for_label_event(l_idx)
                                : e_idx_end;

    if (e_idx < stop_e_idx) {
      const auto step = std::min(batch_size, stop_e_idx - e_idx);
      const auto batch = store->get_batch(e_idx, step);

      encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);
      e_idx += step;
    } else if (l_idx < l_idx_end) {
      const auto [n_id, y_true] = store->get_label_event(l_idx++);
      const auto num_nodes = n_id.size(0);
      std::vector<torch::Tensor> event_preds;

      for (auto i = 0; i < num_nodes; i += node_batch_size) {
        const auto end = std::min(i + node_batch_size, num_nodes);
        const auto n_id_batch = n_id.slice(0, i, end);

        const auto [z] = encoder->forward(n_id_batch);
        event_preds.push_back(decoder->forward(z));
      }

      const auto y_pred = torch::cat(event_preds, 0);
      perf_list.push_back(compute_ndcg(y_pred, y_true));
    }

    util::progress_bar(e_idx - e_idx_start, e_idx_end - e_idx_start, "Valid",
                       start_time);
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
