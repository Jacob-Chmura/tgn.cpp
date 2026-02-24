#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tgn {

struct TGNConfig {
  const std::size_t embedding_dim = 100;
  const std::size_t memory_dim = 100;
  const std::size_t time_dim = 100;
  const std::size_t num_heads = 2;
  const std::size_t num_nbrs = 10;
  const float dropout = 0.1;
};

struct Batch {
  torch::Tensor src;
  torch::Tensor dst;
  torch::Tensor t;
  torch::Tensor msg;
  std::optional<torch::Tensor> neg_dst;
};

enum class NegStrategy {
  None,         // Node Prop or Inference
  Random,       // Link Prop Training (1:1 random negatives)
  PreComputed,  // Link Prop Eval (Uses pre-comptued negatives)
};

struct Split {
  Split() = default;

  Split(std::size_t s, std::size_t e) : start_(s), end_(e) {
    if (end_ < start_) {
      throw std::out_of_range("Invalid split");
    }
  }

  [[nodiscard]] auto start() const -> std::size_t { return start_; }
  [[nodiscard]] auto end() const -> std::size_t { return end_; }
  [[nodiscard]] auto size() const -> std::size_t { return end_ - start_; }

  std::size_t start_{0};
  std::size_t end_{0};
};

class TGStore {
 public:
  virtual ~TGStore() = default;

  [[nodiscard]] virtual auto num_edges() const -> std::size_t = 0;
  [[nodiscard]] virtual auto num_nodes() const -> std::size_t = 0;
  [[nodiscard]] virtual auto msg_dim() const -> std::size_t = 0;

  [[nodiscard]] virtual auto train_split() const -> Split = 0;
  [[nodiscard]] virtual auto val_split() const -> Split = 0;
  [[nodiscard]] virtual auto test_split() const -> Split = 0;

  [[nodiscard]] virtual auto get_batch(
      std::size_t start, std::size_t size,
      NegStrategy strategy = NegStrategy::None) const -> Batch = 0;
  [[nodiscard]] virtual auto gather_timestamps(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;
  [[nodiscard]] virtual auto gather_msgs(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;
};

struct InMemoryTGStoreOptions {
  torch::Tensor src;
  torch::Tensor dst;
  torch::Tensor t;
  torch::Tensor msg;
  std::optional<torch::Tensor> neg_dst = std::nullopt;
  std::optional<std::size_t> val_start = std::nullopt;
  std::optional<std::size_t> test_start = std::nullopt;
};

auto make_store(const InMemoryTGStoreOptions& opts) -> std::shared_ptr<TGStore>;

class TGNImpl : public torch::nn::Module {
 public:
  TGNImpl(const TGNConfig& cfg, const std::shared_ptr<TGStore>& store);
  ~TGNImpl();

  auto detach_memory() -> void;
  auto reset_state() -> void;
  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& t, const torch::Tensor& msg) -> void;

  template <typename... Ts>
  auto forward(const Ts&... inputs) {
    if constexpr (sizeof...(inputs) == 0) {
      throw std::invalid_argument(
          "TGN::forward requires at least one input ID tensor.");
    }
    std::vector<torch::Tensor> input_list = {inputs...};
    auto results = forward_internal(input_list);
    return vec_to_tuple<sizeof...(inputs)>(
        results, std::make_index_sequence<sizeof...(inputs)>{});
  }

 private:
  auto forward_internal(const std::vector<torch::Tensor>& input_list)
      -> std::vector<torch::Tensor>;

  template <std::size_t N, std::size_t... Is>
  auto vec_to_tuple(const std::vector<torch::Tensor>& v,
                    std::index_sequence<Is...>) {
    return std::make_tuple(v[Is]...);
  }

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

TORCH_MODULE(TGN);

}  // namespace tgn
