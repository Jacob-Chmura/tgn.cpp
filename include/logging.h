#pragma once

#include <chrono>
#include <format>
#include <iostream>
#include <string_view>

/** @namespace tguf::detail
 * @brief Internal implementation details for the logging system.
 */
namespace tguf::detail {

/// Severity levels for log filtering and output.
enum class LogLevel {
  DEBUG,
  INFO,
  WARN,
  ERROR,
};

/** * @brief Core logging implementation.
 * Formats and outputs the message to stderr with timestamp and file location.
 */
inline void log_impl(LogLevel level, std::string_view file, int line,
                     std::string_view msg) {
  std::string_view prefix{};

  switch (level) {
    case LogLevel::DEBUG:
      prefix = "[DEBUG]";
      break;
    case LogLevel::WARN:
      prefix = "[WARN ]";
      break;
    case LogLevel::ERROR:
      prefix = "[ERR  ]";
      break;
    case LogLevel::INFO:
    default:  // Default to INFO
      prefix = "[INFO ]";
      break;
  }

  const auto now = std::chrono::floor<std::chrono::milliseconds>(
      std::chrono::system_clock::now());
  const auto pos = file.find_last_of("/\\");
  const auto short_file =
      (pos == std::string_view::npos) ? file : file.substr(pos + 1);
  const auto loc = std::format("{}:{}", short_file, line);
  std::cerr << std::format("{} {:%H:%M:%S} [{:<17}] {}\n", prefix, now, loc,
                           msg);
}

}  // namespace tguf::detail

/// @def TGUF_LOG_DEBUG
/// @brief Logs a debug message. Compiled out in release builds.
#ifndef NDEBUG
#define TGUF_LOG_DEBUG(...)                                                 \
  tguf::detail::log_impl(tguf::detail::LogLevel::DEBUG, __FILE__, __LINE__, \
                         std::format(__VA_ARGS__))
#else
#define TGUF_LOG_DEBUG(...) ((void)0)
#endif

/// @def TGUF_LOG_INFO
/// @brief Logs an informational message.
#define TGUF_LOG_INFO(msg, ...)                                            \
  tguf::detail::log_impl(tguf::detail::LogLevel::INFO, __FILE__, __LINE__, \
                         std::format(msg __VA_OPT__(, ) __VA_ARGS__))

/// @def TGUF_LOG_WARN
/// @brief Logs a warning message for non-critical issues.
#define TGUF_LOG_WARN(msg, ...)                                            \
  tguf::detail::log_impl(tguf::detail::LogLevel::WARN, __FILE__, __LINE__, \
                         std::format(msg __VA_OPT__(, ) __VA_ARGS__))

/// @def TGUF_LOG_ERROR
/// @brief Logs an error message for critical failures.
#define TGUF_LOG_ERROR(msg, ...)                                            \
  tguf::detail::log_impl(tguf::detail::LogLevel::ERROR, __FILE__, __LINE__, \
                         std::format(msg __VA_OPT__(, ) __VA_ARGS__))
