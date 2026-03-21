#pragma once

#include <logging.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

namespace util {

struct TGNArgs {
  std::string tguf_path;

  std::size_t epochs = 10;
  std::size_t batch_size = 200;
  double lr = 1e-4;

  std::size_t embedding_dim = 100;
  std::size_t memory_dim = 100;
  std::size_t time_dim = 100;
  std::size_t num_heads = 2;
  std::size_t num_nbrs = 10;
  float dropout = 0.1f;
};

inline auto parse_args(int argc, char** argv) -> TGNArgs {
  auto print_usage = [argv]() {
    std::cerr << "Usage: " << argv[0] << " <path_to_tguf> [options]\n"
              << "Options:\n"
              << "  --epochs <N>      (default: 10)\n"
              << "  --batch-size <N>  (default: 200)\n"
              << "  --lr <val>        (default: 1e-4)\n"
              << "  --emb-dim <N>     (default: 100)\n"
              << "  --mem-dim <N>     (default: 100)\n"
              << "  --time-dim <N>    (default: 100)\n"
              << "  --heads <N>       (default: 2)\n"
              << "  --nbrs <N>        (default: 10)\n"
              << "  --dropout <val>   (default: 0.1)\n"
              << "  --help            Show this message\n";
    throw std::runtime_error{"Help requested or missing arguments"};
  };

  if (argc < 2) {
    print_usage();
  }
  for (auto i = 1; i < argc; ++i) {
    if (std::string_view{argv[i]} == "--help") {
      print_usage();
    }
  }

  TGNArgs args{};
  args.tguf_path = argv[1];

  auto to_type = []<typename T>(std::string_view val) -> T {
    try {
      if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(std::stod(std::string{val}));
      }
      if constexpr (std::is_signed_v<T>) {
        return static_cast<T>(std::stoll(std::string{val}));
      }
      return static_cast<T>(std::stoul(std::string{val}));
    } catch (...) {
      throw std::runtime_error{std::string{"Invalid numeric value: "} +
                               std::string{val}};
    }
  };

  for (auto i = 2; i < argc - 1; ++i) {
    std::string_view arg{argv[i]};
    std::string_view val{argv[i + 1]};

    if (arg == "--epochs") {
      args.epochs = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--batch-size") {
      args.batch_size = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--lr") {
      args.lr = to_type.template operator()<double>(val);
    } else if (arg == "--emb-dim") {
      args.embedding_dim = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--mem-dim") {
      args.memory_dim = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--time-dim") {
      args.time_dim = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--heads") {
      args.num_heads = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--nbrs") {
      args.num_nbrs = to_type.template operator()<std::size_t>(val);
    } else if (arg == "--dropout") {
      args.dropout = to_type.template operator()<float>(val);
    } else {
      continue;
    }
    i++;
  }

  TGUF_LOG_INFO(" TGUF Path:    {}", args.tguf_path);
  TGUF_LOG_INFO(" Epochs:       {}", args.epochs);
  TGUF_LOG_INFO(" Batch Size:   {}", args.batch_size);
  TGUF_LOG_INFO(" Learning Rate:{:.2e}", args.lr);
  TGUF_LOG_INFO(" Embedding Dim:{}", args.embedding_dim);
  TGUF_LOG_INFO(" Memory Dim:   {}", args.memory_dim);
  TGUF_LOG_INFO(" Time Dim:     {}", args.time_dim);
  TGUF_LOG_INFO(" Num Heads:    {}", args.num_heads);
  TGUF_LOG_INFO(" Neighbors:    {}", args.num_nbrs);
  TGUF_LOG_INFO(" Dropout:      {:.2f}", args.dropout);
  return args;
}

inline auto progress_bar = [](std::size_t current, std::size_t total,
                              std::chrono::steady_clock::time_point start_time,
                              const std::string& prefix = "",
                              const std::string& suffix = "") {
  const auto progress =
      (total == 0) ? 1.0f : static_cast<float>(current) / total;
  const int bar_width = 30;
  const int pos = static_cast<int>(bar_width * progress);

  const auto now = std::chrono::steady_clock::now();
  const double elapsed_sec =
      std::chrono::duration<double>(now - start_time).count();
  const double eps =
      (elapsed_sec > 0.1) ? static_cast<double>(current) / elapsed_sec : 0.0;

  auto format_val = [](std::size_t n) -> std::string {
    if (n >= 1'000'000) {
      return std::format("{:>5.1f}M", n / 1'000'000.0);
    }
    if (n >= 1'000) {
      return std::format("{:>5.1f}K", n / 1'000.0);
    }
    return std::format("{:>6}", n);
  };

  std::cout << "\33[2K\r" << std::left << std::setw(18) << prefix << " ┃";
  for (auto i = 0; i < bar_width; ++i) {
    if (i < pos) {
      std::cout << "█";
    } else if (i == pos) {
      std::cout << "▓";
    } else {
      std::cout << "░";
    }
  }
  std::cout << "┃ ";

  std::cout << std::right << std::setw(3)
            << static_cast<int>(std::round(progress * 100.0)) << "% "
            << "(" << format_val(current) << " / " << format_val(total) << ")";

  auto total_sec = static_cast<std::int64_t>(std::round(elapsed_sec));
  std::cout << " │ Time: " << std::setfill('0') << std::setw(2)
            << (total_sec / 60) << ":" << std::setfill('0') << std::setw(2)
            << (total_sec % 60) << std::setfill(' ');

  std::cout << " │ " << std::fixed << std::setprecision(1) << std::setw(6);
  if (eps >= 1'000'000) {
    std::cout << eps / 1'000'000.0 << "M edges/s";
  } else {
    std::cout << eps / 1'000.0 << "K edges/s";
  }

  if (!suffix.empty()) {
    std::cout << " │ " << suffix;
  }
  std::cout << std::flush;
};

inline auto log_torch_backend_info() -> void {
  TGUF_LOG_INFO("LibTorch Backend | Intra-op threads: {}",
                torch::get_num_threads());
  TGUF_LOG_INFO("LibTorch Backend | Inter-op threads: {}",
                torch::get_num_interop_threads());
  TGUF_LOG_INFO("LibTorch Backend | CPU Capability: {}",
                torch::get_cpu_capability());

#ifdef AT_MKL_ENABLED
  TGUF_LOG_INFO("LibTorch Backend | BLAS: Intel MKL (Enabled)");
#else
  TGUF_LOG_WARN("LibTorch Backend | BLAS: Intel MKL not found");
#endif
};

}  // namespace util
