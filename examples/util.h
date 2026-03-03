#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace util {
inline auto progress_bar =
    [](std::size_t current, std::size_t total, const std::string& prefix,
       std::chrono::steady_clock::time_point start_time) {
      const auto progress =
          static_cast<float>(current) / static_cast<float>(total);
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

}  // namespace util
