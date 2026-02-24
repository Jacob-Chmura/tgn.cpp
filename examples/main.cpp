#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

#include <chrono>
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
constexpr std::size_t batch_size = 200;
constexpr double learning_rate = 1e-5;

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

auto progress_bar = [](std::size_t current, std::size_t total,
                       const std::string& prefix,
                       std::chrono::steady_clock::time_point start_time) {
  const auto progress = static_cast<float>(current) / static_cast<float>(total);
  const auto bar_width = 30;
  const int pos = bar_width * progress;

  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
          .count();
  const auto minutes = elapsed / 60;
  const auto seconds = elapsed % 60;

  std::cout << "\r" << prefix << " [";
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos) {
      std::cout << "=";
    } else if (i == pos) {
      std::cout << ">";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0)
            << "% | " << std::setfill('0') << std::setw(2) << minutes << ":"
            << std::setfill('0') << std::setw(2) << seconds << std::flush;
};

auto train(tgn::TGN& encoder, LinkPredictor& decoder, torch::optim::Adam& opt,
           const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();
  encoder->train();
  decoder->train();
  encoder->reset_state();

  float total_loss{0};
  auto e_id = store->train_split().start;

  for (; e_id < store->train_split().end; e_id += batch_size) {
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

    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);
    loss.backward();
    opt.step();
    encoder->detach_memory();

    total_loss += loss.item<float>();

    progress_bar(e_id - store->train_split().start, store->train_split().size(),
                 "Train", start_time);
  }
  std::cout << std::endl;
  return total_loss / static_cast<float>(store->train_split().size());
}

auto eval(tgn::TGN& encoder, LinkPredictor& decoder,
          const std::shared_ptr<tgn::TGStore>& store) -> float {
  auto start_time = std::chrono::steady_clock::now();
  encoder->eval();
  decoder->eval();

  float mrr{0};
  auto e_id = store->val_split().start;

  torch::NoGradGuard no_grad;

  for (; e_id < store->val_split().end; e_id += batch_size) {
    const auto batch =
        store->get_batch(e_id, batch_size, tgn::NegStrategy::Fixed);
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

    // TODO(kuba): mrr implementation

    encoder->update_state(batch.src, batch.dst, batch.t, batch.msg);

    progress_bar(e_id - store->val_split().start, store->val_split().size(),
                 "Valid", start_time);
  }
  std::cout << std::endl;
  return mrr;
}

auto load_csv(const std::string& path) -> tgn::InMemoryTGStoreOptions {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  std::string line{};
  std::size_t val_start = 0;
  std::size_t test_start = 0;
  auto found_metadata{false};

  // Parse Metadata Line (# val_start:123,test_start:456)
  if (std::getline(file, line)) {
    if (line.size() > 1 && line[0] == '#') {
      try {
        const auto v_pos = line.find("val_start:");
        const auto comma_pos = line.find(",");
        const auto t_pos = line.find("test_start:");

        if (v_pos != std::string::npos && t_pos != std::string::npos) {
          const auto v_val_start = v_pos + 10;  // length of "val_start:"
          const auto t_val_start = t_pos + 11;  // length of "test_start:"

          val_start =
              std::stoull(line.substr(v_val_start, comma_pos - v_val_start));
          test_start = std::stoull(line.substr(t_val_start));
          found_metadata = true;
        }
      } catch (const std::exception& e) {
        throw std::runtime_error("Malformed metadata header in CSV: " +
                                 std::string(e.what()));
      }
    }
  }

  if (!found_metadata) {
    throw std::runtime_error("CSV missing required split metadata header");
  }
  if (!std::getline(file, line)) {
    throw std::runtime_error("CSV file is empty or missing column headers");
  }

  std::stringstream header_ss(line);
  std::string col_name{};
  std::vector<std::string> headers;
  std::size_t msg_start_idx{0};
  std::size_t neg_start_idx{0};
  std::size_t idx{0};

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

  // If no negatives are found, set index to end of headers
  if (neg_start_idx == 0) {
    neg_start_idx = headers.size();
  }

  const std::size_t msg_dim =
      (msg_start_idx == 0) ? 0 : (neg_start_idx - msg_start_idx);
  const std::size_t neg_dim = headers.size() - neg_start_idx;

  std::vector<std::int64_t> src_vec;
  std::vector<std::int64_t> dst_vec;
  std::vector<std::int64_t> t_vec;
  std::vector<std::int64_t> neg_vec;
  std::vector<float> msg_vec;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
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
      } else if (msg_dim > 0 && curr_col >= msg_start_idx &&
                 curr_col < neg_start_idx) {
        msg_vec.push_back(std::stof(cell));
      } else if (neg_dim > 0 && curr_col >= neg_start_idx) {
        neg_vec.push_back(std::stoll(cell));
      }
      curr_col++;
    }
  }

  const auto n = static_cast<std::int64_t>(src_vec.size());

  std::cout << "Loaded " << n << " edges (val_start: " << val_start
            << ", test_start: " << test_start << ")" << std::endl;

  return tgn::InMemoryTGStoreOptions{
      .src = torch::tensor(src_vec, torch::kLong),
      .dst = torch::tensor(dst_vec, torch::kLong),
      .t = torch::tensor(t_vec, torch::kLong),
      .msg = (msg_dim > 0) ? torch::tensor(msg_vec).view(
                                 {n, static_cast<std::int64_t>(msg_dim)})
                           : torch::empty({n, 0}),
      .neg_dst = (neg_dim > 0)
                     ? std::optional<torch::Tensor>(
                           torch::tensor(neg_vec, torch::kLong).view({n, -1}))
                     : std::nullopt,
      .val_start = val_start,
      .test_start = test_start,
  };
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
    auto mrr = eval(encoder, decoder, store);
    std::cout << "Epoch " << epoch << " Loss: " << loss << " MRR: " << mrr
              << std::endl;
  }
}
