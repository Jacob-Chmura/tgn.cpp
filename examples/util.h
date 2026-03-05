#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

namespace util {
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
}  // namespace util
