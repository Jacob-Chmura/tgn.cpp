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

#include "logging.h"
#include "tgn.h"
#include "util.h"

constexpr std::size_t num_epochs = 10;
constexpr double learning_rate = 1e-4;

namespace {

std::size_t current_epoch = 1;

struct NodePredictorImpl : torch::nn::Module {
  explicit NodePredictorImpl(std::size_t in_dim, std::size_t out_dim,
                             std::size_t hidden_dim = 64) {
    model_ = torch::nn::Sequential(torch::nn::Linear(in_dim, hidden_dim),
                                   torch::nn::ReLU(),
                                   torch::nn::Linear(hidden_dim, out_dim));
    register_module("model_", model_);
    TGN_LOG_INFO(
        "NodeDecoder: Initialized (in_channels={}, hidden_dim={}, "
        "out_channels={})",
        in_dim, hidden_dim, out_dim);
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
  const auto ranks = torch::arange(1, k + 1).to(torch::kFloat32);
  const auto discounts = torch::log2(ranks + 1.0);

  const auto [pred_labels, pred_indices] = y_pred.topk(k, -1);
  const auto y_true_at_pred_topk = y_true.gather(-1, pred_indices);
  const auto dcg = (y_true_at_pred_topk / discounts).sum(-1);

  const auto [ideal_labels, ideal_indices] = y_true.topk(k, -1);
  const auto idcg = (ideal_labels / discounts).sum(-1);

  const auto ndcg = dcg / (idcg + 1e-8);
  return ndcg.mean().item<float>();
}

auto train(tgn::TGN& encoder, NodePredictor& decoder, torch::optim::Adam& opt,
           const std::shared_ptr<tgn::TGStore>& store) -> void {
  auto start_time = std::chrono::steady_clock::now();
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};

  const auto e_range = store->train_split();
  const auto l_range = store->train_label_split();
  auto e_id = e_range.start();
  auto l_id = l_range.start();

  while (l_id < l_range.end()) {
    // Catch up all edge events before current label event
    const auto stop_e_id = store->get_stop_e_id_for_label_event(l_id);
    if (e_id < stop_e_id) {
      const auto num_edges_to_process = stop_e_id - e_id;
      const auto batch = store->get_batch(e_id, num_edges_to_process);

      encoder->update_state(batch.src, batch.dst, batch.time, batch.msg);
      e_id = stop_e_id;
    }

    opt.zero_grad();

    const auto label_event = store->get_label_event(l_id++);
    const auto [z] = encoder->forward(label_event.n_id);
    const auto y_pred = decoder->forward(z);

    auto loss =
        torch::nn::functional::cross_entropy(y_pred, label_event.y_true);
    loss.backward();
    opt.step();
    total_loss += loss.item<float>();

    encoder->detach_memory();

    util::progress_bar(
        e_id - e_range.start(), e_range.size(), start_time,
        std::format("Epoch {:2d}/{:2d} [Train]", current_epoch, num_epochs),
        std::format("Loss: {:.3f}",
                    total_loss / static_cast<float>(std::max<std::size_t>(
                                     1, l_id - l_range.start()))));
  }
  std::cout << std::endl;
}

auto eval(tgn::TGN& encoder, NodePredictor& decoder,
          const std::shared_ptr<tgn::TGStore>& store) -> void {
  auto start_time = std::chrono::steady_clock::now();

  torch::NoGradGuard no_grad;
  encoder->eval();
  decoder->eval();

  std::vector<float> perf_list;

  const auto e_range = store->val_split();
  const auto l_range = store->val_label_split();
  auto e_id = e_range.start();
  auto l_id = l_range.start();

  while (l_id < l_range.end()) {
    const auto stop_e_id = store->get_stop_e_id_for_label_event(l_id);
    if (e_id < stop_e_id) {
      const auto num_edges_to_process = stop_e_id - e_id;
      const auto batch = store->get_batch(e_id, num_edges_to_process);

      encoder->update_state(batch.src, batch.dst, batch.time, batch.msg);
      e_id = stop_e_id;
    }

    const auto label_event = store->get_label_event(l_id++);
    const auto [z] = encoder->forward(label_event.n_id);
    const auto y_pred = decoder->forward(z);
    perf_list.push_back(compute_ndcg(y_pred, label_event.y_true));

    util::progress_bar(
        e_id - e_range.start(), e_range.size(), start_time,
        std::format("            [Valid]", current_epoch, num_epochs),
        std::format("NDCG@10: {:.3f}",
                    std::accumulate(perf_list.begin(), perf_list.end(), 0.0F) /
                        static_cast<float>(perf_list.size())));
  }
  std::cout << std::endl;
}

}  // namespace

auto main(int argc, char** argv) -> int {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_tguf>" << std::endl;
    return 1;
  }
  const std::string tguf_path = argv[1];
  TGN_LOG_INFO("Running Node Property Prediction ({})", tguf_path);

  tgn::TGUFOptions opts{.path = tguf_path};
  const auto store = tgn::TGStore::from_tguf(opts);
  const auto cfg = tgn::TGNConfig{};

  tgn::TGN encoder(cfg, store);
  const auto num_classes = store->label_dim();
  NodePredictor decoder{cfg.embedding_dim, num_classes};

  auto params = encoder->parameters();
  auto dec_params = decoder->parameters();
  params.insert(params.end(), dec_params.begin(), dec_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(learning_rate));

  while (current_epoch <= num_epochs) {
    train(encoder, decoder, opt, store);
    eval(encoder, decoder, store);
    ++current_epoch;
  }
}
