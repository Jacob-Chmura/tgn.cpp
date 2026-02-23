#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "lib.h"

constexpr std::size_t num_epochs = 10;
constexpr std::size_t batch_size = 5;
constexpr double learning_rate = 1e-3;

namespace {

struct LinkPredictorImpl : torch::nn::Module {
  explicit LinkPredictorImpl(std::size_t in_channels) {
    w_src_ =
        register_module("w_src_", torch::nn::Linear(in_channels, in_channels));
    w_dst_ =
        register_module("w_dst_", torch::nn::Linear(in_channels, in_channels));
    w_final_ = register_module("w_final_", torch::nn::Linear(in_channels, 1));
  }

  auto forward(const torch::Tensor& z_src, const torch::Tensor& z_dst)
      -> torch::Tensor {
    const auto z = torch::relu(w_src_->forward(z_src) + w_dst_->forward(z_dst));
    return w_final_->forward(z);
  }

 private:
  torch::nn::Linear w_src_{nullptr}, w_dst_{nullptr}, w_final_{nullptr};
};
TORCH_MODULE(LinkPredictor);

auto train(tgn::TGN& encoder, LinkPredictor& decoder, torch::optim::Adam& opt,
           const std::shared_ptr<tgn::TGStore>& store) -> float {
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};
  std::size_t e_id = 0;

  for (; e_id < store->num_edges(); e_id += batch_size) {
    opt.zero_grad();

    const auto batch = store->get_batch(e_id, batch_size);
    const auto [z_src, z_dst, z_neg] =
        encoder->forward(batch.src, batch.dst, batch.neg_dst->flatten());

    const auto pos_out = decoder->forward(z_src, z_dst);

    // Pair each src with its M negatives for decoding
    // Expand src [N, D] -> [N, M, D] then flatten both to [N*M, D]
    const auto N = z_src.size(0);
    const auto D = z_src.size(1);
    const auto M = batch.neg_dst->size(1);
    const auto z_src_expanded =
        z_src.unsqueeze(1).expand({N, M, D}).reshape({-1, D});
    const auto neg_out =
        decoder->forward(z_src_expanded, z_neg.reshape({-1, D}));

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

auto load_csv(const std::string& path) -> tgn::InMemoryTGStoreOptions {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  std::string line{};
  std::string col_name{};
  std::getline(file, line);
  std::stringstream header_ss(line);

  std::vector<std::string> headers;
  std::size_t msg_start_idx = 0;
  std::size_t neg_start_idx = 0;
  std::size_t idx = 0;

  while (std::getline(header_ss, col_name, ',')) {
    headers.push_back(col_name);
    if (msg_start_idx == 0 && col_name.find("msg") != std::string::npos) {
      msg_start_idx = idx;
    }
    if (neg_start_idx == 0 && col_name.find("neg") != std::string::npos) {
      neg_start_idx = idx;
    }
    idx++;
  }

  if (neg_start_idx == 0) {
    neg_start_idx = headers.size();
  }

  const std::size_t msg_dim = neg_start_idx - msg_start_idx;
  const std::size_t neg_dim = headers.size() - neg_start_idx;

  std::vector<std::int64_t> src_vec;
  std::vector<std::int64_t> dst_vec;
  std::vector<std::int64_t> t_vec;
  std::vector<std::int64_t> neg_vec;
  std::vector<float> msg_vec;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell{};
    std::size_t curr_col{0};

    while (std::getline(ss, cell, ',')) {
      if (curr_col == 0) {
        src_vec.push_back(std::stoll(cell));
      } else if (curr_col == 1) {
        dst_vec.push_back(std::stoll(cell));
      } else if (curr_col == 2) {
        t_vec.push_back(std::stoll(cell));
      } else if (curr_col >= msg_start_idx && curr_col < neg_start_idx) {
        msg_vec.push_back(std::stof(cell));
      } else if (curr_col >= neg_start_idx) {
        neg_vec.push_back(std::stoll(cell));
      }
      curr_col++;
    }
  }

  auto n = static_cast<std::int64_t>(src_vec.size());
  std::cout << "Loaded " << n << " edges from " << path
            << " (msg_dim: " << msg_dim << ", num_negatives: " << neg_dim << ")"
            << std::endl;

  return tgn::InMemoryTGStoreOptions{
      .src = torch::tensor(src_vec, torch::kLong),
      .dst = torch::tensor(dst_vec, torch::kLong),
      .t = torch::tensor(t_vec, torch::kLong),
      .msg =
          torch::tensor(msg_vec).view({n, static_cast<std::int64_t>(msg_dim)}),
      .neg_dst = neg_dim > 0
                     ? std::optional<torch::Tensor>(
                           torch::tensor(neg_vec, torch::kLong).view({n, -1}))
                     : std::nullopt};
}

}  // namespace

auto main() -> int {
  const auto cfg = tgn::TGNConfig{};
  const auto opts = load_csv("data/tgbl-wiki.csv");
  const auto store = tgn::make_store(opts);

  tgn::TGN encoder(cfg, store);
  LinkPredictor decoder{cfg.embedding_dim};

  auto params = encoder->parameters();
  auto dec_params = decoder->parameters();
  params.insert(params.end(), dec_params.begin(), dec_params.end());
  torch::optim::Adam opt(params, torch::optim::AdamOptions(learning_rate));

  for (std::size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    auto loss = train(encoder, decoder, opt, store);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
  }
}
