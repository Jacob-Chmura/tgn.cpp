#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace util {
inline auto progress_bar = [](std::size_t current, std::size_t total,
                              std::chrono::steady_clock::time_point start_time,
                              const std::string& prefix = "",
                              const std::string& suffix = "") {
  const auto progress =
      (total == 0) ? 0.0f
                   : static_cast<float>(current) / static_cast<float>(total);
  const int bar_width = 40;
  const int pos = static_cast<int>(bar_width * progress);

  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
          .count();

  // \33[2K clears the line, \r returns cursor to the start
  std::cout << "\33[2K\r" << prefix << " ┃";
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos)
      std::cout << "█";
    else if (i == pos)
      std::cout << "▓";
    else
      std::cout << "░";
  }

  std::cout << "┃ " << std::right << std::setw(3)
            << static_cast<int>(progress * 100.0) << "%"
            << " │ Time: " << std::setfill('0') << std::setw(2) << elapsed / 60
            << ":" << std::setw(2) << elapsed % 60 << std::setfill(' ');

  if (!suffix.empty()) {
    std::cout << " | " << suffix;
  }

  std::cout << std::flush;
};

}  // namespace util
