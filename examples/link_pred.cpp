#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tgn.h"
#include "util.h"

constexpr std::size_t num_epochs = 10;
constexpr std::size_t batch_size = 200;
constexpr double learning_rate = 1e-4;
constexpr std::string dataset = "tgbl-wiki";

namespace {

struct LinkPredictorImpl : torch::nn::Module {
  explicit LinkPredictorImpl(std::size_t in_dim) {
    w_src_ = register_module("w_src_", torch::nn::Linear(in_dim, in_dim));
    w_dst_ = register_module("w_dst_", torch::nn::Linear(in_dim, in_dim));
    w_final_ = register_module("w_final_", torch::nn::Linear(in_dim, 1));
  }

  auto forward(const torch::Tensor& z_src, const torch::Tensor& z_dst)
      -> torch::Tensor {
    const auto z = torch::relu(w_src_->forward(z_src) + w_dst_->forward(z_dst));
    return w_final_->forward(z).view(-1);
  }

 private:
  torch::nn::Linear w_src_{nullptr}, w_dst_{nullptr}, w_final_{nullptr};
};
TORCH_MODULE(LinkPredictor);

auto compute_mrr(const torch::Tensor& pred_pos, const torch::Tensor& pred_neg)
    -> float {
  const auto n = pred_pos.size(0);
  const auto m = pred_neg.size(0) / n;

  const auto y_pred_pos = pred_pos.view({n, 1});
  const auto y_pred_neg = pred_neg.view({n, m});

  const auto optimistic_rank = (y_pred_neg > y_pred_pos).sum(1);
  const auto pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(1);
  const auto ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1.0;
  const auto mrr_list = 1.0 / ranking_list.to(torch::kFloat32);
  return mrr_list.mean().item<float>();
}

auto train(tgn::TGN& encoder, LinkPredictor& decoder, torch::optim::Adam& opt,
           const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};
  const auto e_range = store->train_split();

  for (auto e_id = e_range.start(); e_id < e_range.end(); e_id += batch_size) {
    opt.zero_grad();

    const auto batch =
        store->get_batch(e_id, batch_size, tgn::NegStrategy::Random);
    const auto [z_src, z_dst, z_neg] =
        encoder->forward(batch.src, batch.dst, batch.neg_dst->flatten());

    // Assumes training negatives are 1:1 with positives
    const auto pos_out = decoder->forward(z_src, z_dst);
    const auto neg_out = decoder->forward(z_src, z_neg);

    auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    pos_out, torch::ones_like(pos_out)) +
                torch::nn::functional::binary_cross_entropy_with_logits(
                    neg_out, torch::zeros_like(neg_out));
    loss.backward();
    opt.step();
    total_loss += loss.item<float>();

    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);
    encoder->detach_memory();

    util::progress_bar(e_id - e_range.start(), e_range.size(), "Train",
                       start_time);
  }

  std::cout << std::endl;
  return total_loss / static_cast<float>(e_range.size());
}

auto eval(tgn::TGN& encoder, LinkPredictor& decoder,
          const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();

  torch::NoGradGuard no_grad;
  encoder->eval();
  decoder->eval();

  std::vector<float> perf_list;
  const auto e_range = store->val_split();

  for (auto e_id = e_range.start(); e_id < e_range.end(); e_id += batch_size) {
    const auto batch =
        store->get_batch(e_id, batch_size, tgn::NegStrategy::PreComputed);
    const auto [z_src, z_dst, z_neg] =
        encoder->forward(batch.src, batch.dst, batch.neg_dst->flatten());

    const auto pred_pos = decoder->forward(z_src, z_dst).sigmoid();

    // Pair each src with its M negatives for decoding
    // Expand src [N, D] -> [N, M, D] then flatten both to [N*M, D]
    const auto N = z_src.size(0);
    const auto D = z_src.size(1);
    const auto M = batch.neg_dst->size(1);
    const auto z_src_expanded =
        z_src.unsqueeze(1).expand({N, M, D}).reshape({-1, D});
    const auto pred_neg =
        decoder->forward(z_src_expanded, z_neg.reshape({-1, D})).sigmoid();

    perf_list.push_back(compute_mrr(pred_pos, pred_neg));
    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);

    util::progress_bar(e_id - e_range.start(), e_range.size(), "Valid",
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
  auto data = util::load_csv("data/" + dataset);

  std::shared_ptr<tgn::TGStore> store;
  const auto use_tguf = true;

  if (use_tguf) {
    // This should be done ahead of time via our CLI
    // const std::string tguf_path = "data/" + dataset + ".tguf";
    const std::string tguf_path = "fooo_wiki.tguf";
    // std::cout << "Building TGUF file: " << tguf_path << "..." << std::endl;

    // const std::size_t n_edges = data.src.size(0);
    // const std::size_t m_dim = data.msg.size(1);
    // const std::size_t n_neg = data.neg_dst->size(1);
    // const std::size_t n_labels = 0;
    // const std::size_t l_dim = 0;

    // tgn::TGUFBuilder builder(tguf_path, n_edges, n_labels, m_dim, l_dim,
    // n_neg);

    // tgn::Batch batch{.src = data.src,
    //                  .dst = data.dst,
    //                  .t = data.t,
    //                  .msg = data.msg,
    //                  .neg_dst = data.neg_dst};
    // builder.append_edges(batch);
    // builder.finalize();
    // std::cout << "TGUF construction complete." << std::endl;

    std::cout << "Loading store from tguf (mmap)..." << std::endl;
    tgn::TGUFOptions opts{.path = tguf_path,
                          .val_start = data.val_start,
                          .test_start = data.test_start};
    store = tgn::TGStore::from_tguf(opts);
  } else {
    std::cout << "Loading store from memory..." << std::endl;
    store = tgn::TGStore::from_memory(std::move(data));
  }

  tgn::TGN encoder(cfg, store);
  LinkPredictor decoder{cfg.embedding_dim};

  auto params = encoder->parameters();
  auto dec_params = decoder->parameters();
  params.insert(params.end(), dec_params.begin(), dec_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(learning_rate));

  for (std::size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    auto loss = train(encoder, decoder, opt, store);
    auto mrr = eval(encoder, decoder, store);
    std::cout << "Epoch " << epoch << " Loss: " << loss << " MRR: " << mrr
              << std::endl;
  }
}
