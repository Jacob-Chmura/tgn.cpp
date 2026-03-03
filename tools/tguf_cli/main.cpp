#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "tgn.h"

namespace {
struct TGUFConfig {
  std::string out_path;
  std::size_t n_edges{};
  std::size_t m_dim{};
  std::size_t n_neg{};
  std::size_t n_labels{};
  std::size_t l_dim{};
  std::size_t val_start{};
  std::size_t test_start{};
};

auto parse_args(int argc, char** argv) -> TGUFConfig {
  TGUFConfig conf;
  for (auto i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      conf.out_path = argv[++i];
    } else if (arg == "--val_start" && i + 1 < argc) {
      conf.val_start = std::stoull(argv[++i]);
    } else if (arg == "--test_start" && i + 1 < argc) {
      conf.test_start = std::stoull(argv[++i]);
    } else if (arg == "--n_edges" && i + 1 < argc) {
      conf.n_edges = std::stoull(argv[++i]);
    } else if (arg == "--m_dim" && i + 1 < argc) {
      conf.m_dim = std::stoull(argv[++i]);
    } else if (arg == "--n_neg" && i + 1 < argc) {
      conf.n_neg = std::stoull(argv[++i]);
    } else if (arg == "--n_labels" && i + 1 < argc) {
      conf.n_labels = std::stoull(argv[++i]);
    } else if (arg == "--l_dim" && i + 1 < argc) {
      conf.l_dim = std::stoull(argv[++i]);
    }
  }

  if (conf.out_path.empty() || (conf.n_edges == 0 && conf.n_labels == 0)) {
    std::cerr << "Usage: tguf_cli --out <path> --n_edges <N> --m_dim <D> "
                 "[--n_neg <K>] [--n_labels <L> --l_dim <D2>]\n";
    exit(1);
  }
  return conf;
}

auto read_exactly(void* ptr, std::size_t n_bytes) -> void {
  if (!std::cin.read(static_cast<char*>(ptr), n_bytes)) {
    if (std::cin.eof()) {
      return;
    }
    throw std::runtime_error("Failed to read " + std::to_string(n_bytes) +
                             " bytes from stdin");
  }
}

auto process_edge_batch(const TGUFConfig& conf, tgn::TGUFBuilder& builder,
                        std::int64_t batch_size) -> void {
  std::vector<std::int64_t> src(batch_size);
  std::vector<std::int64_t> dst(batch_size);
  std::vector<std::int64_t> t(batch_size);
  std::vector<float> msg(batch_size * conf.m_dim);

  read_exactly(src.data(), batch_size * sizeof(std::int64_t));
  read_exactly(dst.data(), batch_size * sizeof(std::int64_t));
  read_exactly(t.data(), batch_size * sizeof(std::int64_t));
  read_exactly(msg.data(), batch_size * conf.m_dim * sizeof(float));

  tgn::Batch batch{
      .src = torch::from_blob(src.data(), {batch_size}, torch::kInt64).clone(),
      .dst = torch::from_blob(dst.data(), {batch_size}, torch::kInt64).clone(),
      .t = torch::from_blob(t.data(), {batch_size}, torch::kInt64).clone(),
      .msg = torch::from_blob(msg.data(), {batch_size, conf.m_dim},
                              torch::kFloat32)
                 .clone()};

  if (conf.n_neg > 0) {
    auto neg_dst = std::vector<std::int64_t>(batch_size * conf.n_neg);
    read_exactly(neg_dst.data(),
                 batch_size * conf.n_neg * sizeof(std::int64_t));
    batch.neg_dst = torch::from_blob(neg_dst.data(), {batch_size, conf.n_neg},
                                     torch::kInt64)
                        .clone();
  }
  builder.append_edges(batch);
}

auto process_label_batch(const TGUFConfig& conf, tgn::TGUFBuilder& builder,
                         std::int64_t batch_size) -> void {
  std::vector<std::int64_t> n_id(batch_size);
  std::vector<std::int64_t> t(batch_size);
  std::vector<float> y_true(batch_size * conf.l_dim);

  read_exactly(n_id.data(), batch_size * sizeof(std::int64_t));
  read_exactly(t.data(), batch_size * sizeof(std::int64_t));
  read_exactly(y_true.data(), batch_size * conf.l_dim * sizeof(float));

  builder.append_labels(
      torch::from_blob(n_id.data(), {batch_size}, torch::kInt64).clone(),
      torch::from_blob(t.data(), {batch_size}, torch::kInt64).clone(),
      torch::from_blob(y_true.data(), {batch_size, conf.l_dim}, torch::kFloat32)
          .clone());
}

}  // namespace

auto main(int argc, char** argv) -> int {
  try {
    auto conf = parse_args(argc, argv);
    tgn::TGUFBuilder builder(conf.out_path, conf.n_edges, conf.n_labels,
                             conf.m_dim, conf.l_dim, conf.n_neg, conf.val_start,
                             conf.test_start);
    char cmd;
    std::int64_t batch_size;
    while (std::cin.read(&cmd, 1)) {
      read_exactly(&batch_size, sizeof(std::int64_t));
      if (batch_size <= 0) {
        continue;
      }

      if (cmd == 'E') {
        process_edge_batch(conf, builder, batch_size);
      } else if (cmd == 'L') {
        process_label_batch(conf, builder, batch_size);
      }
    }
    std::cout << "Finalizing builder...\n";
    builder.finalize();

    std::cout << "TGUF construction complete." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error in tguf_cli: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
