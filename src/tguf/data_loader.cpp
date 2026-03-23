#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <utility>

#include "logging.h"
#include "tguf.h"

namespace tguf {

template <typename T>
AsyncDataLoader<T>::AsyncDataLoader(std::size_t prefetch_factor)
    : prefetch_factor_(prefetch_factor) {}

template <typename T>
AsyncDataLoader<T>::~AsyncDataLoader() {
  stop();
}

template <typename T>
template <typename Producer>
auto AsyncDataLoader<T>::start(std::size_t start_idx, std::size_t end_idx,
                               std::size_t batch_size, Producer&& producer)
    -> void {
  stop_ = false;
  worker_ = std::thread([this, start_idx, end_idx, batch_size,
                         fn = std::forward<Producer>(producer)]() mutable {
    for (auto i = start_idx; i < end_idx; i += batch_size) {
      auto current_batch_size = std::min(batch_size, end_idx - i);

      // Wait for space in the prefetch buffer
      std::unique_lock<std::mutex> lock(mutex_);
      cv_full_.wait(lock,
                    [this] { return q_.size() < prefetch_factor_ || stop_; });
      if (stop_) {
        break;
      }

      // Launch the task. We pass 'fn' by value into the async lambda.
      auto task = std::async(std::launch::async, [fn, i, current_batch_size] {
        return fn(i, current_batch_size);
      });

      q_.push(std::move(task));

      // Signal the consumer
      lock.unlock();
      cv_empty_.notify_one();
    }
  });
}

template <typename T>
auto AsyncDataLoader<T>::stop() -> void {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_) {
      return;
    }
    stop_ = true;
  }

  cv_full_.notify_all();
  cv_empty_.notify_all();

  if (worker_.joinable()) {
    worker_.join();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  while (!q_.empty()) {
    q_.pop();
  }
}

template <typename T>
auto AsyncDataLoader<T>::next() -> std::optional<T> {
  std::unique_lock<std::mutex> lock(mutex_);

  // Wait for a task to be available or for the loader to stop
  cv_empty_.wait(lock, [this] { return !q_.empty() || stop_; });
  if (q_.empty()) {
    return std::nullopt;
  }

  // Move the future out of the queue
  auto fut = std::move(q_.front());
  q_.pop();

  // Unlock before blocking on .get() to allow the producer to continue
  lock.unlock();
  cv_full_.notify_one();
  return fut.get();
}

}  // namespace tguf
