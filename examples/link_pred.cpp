#include <torch/torch.h>

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
#include "tguf.h"
#include "util.h"

namespace {

util::TGNArgs args{};
std::size_t current_epoch = 1;

struct LinkPredictorImpl : torch::nn::Module {
  explicit LinkPredictorImpl(std::size_t in_dim) {
    w_src_ = register_module("w_src_", torch::nn::Linear(in_dim, in_dim));
    w_dst_ = register_module("w_dst_", torch::nn::Linear(in_dim, in_dim));
    w_final_ = register_module("w_final_", torch::nn::Linear(in_dim, 1));
    TGN_LOG_INFO("LinkDecoder: Initialized (in_channels={})", in_dim);
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
           const std::shared_ptr<tguf::TGStore>& store) -> void {
  auto start_time = std::chrono::steady_clock::now();
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};
  const auto e_range = store->train_split();

  for (auto e_id = e_range.start(); e_id < e_range.end();
       e_id += args.batch_size) {
    opt.zero_grad();

    const auto batch = store->get_batch(e_id, args.batch_size,
                                        tguf::TGStore::NegStrategy::Random);
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

    encoder->update_state(batch.src, batch.dst, batch.time, batch.msg);
    encoder->detach_memory();

    util::progress_bar(
        e_id - e_range.start(), e_range.size(), start_time,
        std::format("Epoch {:2d}/{:2d} [Train]", current_epoch, args.epochs),
        std::format("Loss: {:.3f}",
                    total_loss / static_cast<float>(std::max<std::size_t>(
                                     1, e_id - e_range.start()))));
  }
  std::cout << std::endl;
}

auto eval(tgn::TGN& encoder, LinkPredictor& decoder,
          const std::shared_ptr<tguf::TGStore>& store) -> void {
  auto start_time = std::chrono::steady_clock::now();

  torch::NoGradGuard no_grad;
  encoder->eval();
  decoder->eval();

  std::vector<float> perf_list;
  const auto e_range = store->val_split();

  for (auto e_id = e_range.start(); e_id < e_range.end();
       e_id += args.batch_size) {
    const auto batch = store->get_batch(
        e_id, args.batch_size, tguf::TGStore::NegStrategy::PreComputed);
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
    encoder->update_state(batch.src, batch.dst, batch.time, batch.msg);

    util::progress_bar(
        e_id - e_range.start(), e_range.size(), start_time,
        std::format("            [Valid]", current_epoch, args.epochs),
        std::format("MRR:  {:.3f}",
                    std::accumulate(perf_list.begin(), perf_list.end(), 0.0F) /
                        static_cast<float>(perf_list.size())));
  }
  std::cout << std::endl;
}

}  // namespace

auto main(int argc, char** argv) -> int {
  TGN_LOG_INFO("Running Link Prediction");
  if (torch::cuda::is_available()) {
    TGN_LOG_INFO("CUDA is available!");
    torch::Device device(torch::kCUDA);
  } else {
    TGN_LOG_INFO("CUDA not found!");
  }
  args = util::parse_args(argc, argv);
  util::log_torch_backend_info();

  const auto store = tguf::TGStore::from_tguf(args.tguf_path);
  const auto cfg = tgn::TGNConfig{.embedding_dim = args.embedding_dim,
                                  .memory_dim = args.memory_dim,
                                  .time_dim = args.time_dim,
                                  .num_heads = args.num_heads,
                                  .num_nbrs = args.num_nbrs,
                                  .dropout = args.dropout};
  tgn::TGN encoder(cfg, store);
  LinkPredictor decoder{cfg.embedding_dim};

  auto params = encoder->parameters();
  auto dec_params = decoder->parameters();
  params.insert(params.end(), dec_params.begin(), dec_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(args.lr));

  while (current_epoch <= args.epochs) {
    train(encoder, decoder, opt, store);
    eval(encoder, decoder, store);
    ++current_epoch;
  }
}
