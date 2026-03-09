#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tgn {

struct TGNConfig {
  std::size_t embedding_dim = 100;
  std::size_t memory_dim = 100;
  std::size_t time_dim = 100;
  std::size_t num_heads = 2;
  std::size_t num_nbrs = 10;
  float dropout = 0.1;
};

struct Batch {
  torch::Tensor src;
  torch::Tensor dst;
  torch::Tensor time;
  torch::Tensor msg;
  std::optional<torch::Tensor> neg_dst;
};

struct LabelEvent {
  torch::Tensor n_id;
  torch::Tensor target;
};

struct TGUFSchema {
  std::string path;

  std::size_t edge_capacity;
  std::size_t label_capacity;
  std::size_t msg_dim;
  std::size_t label_dim;
  std::size_t negatives_capacity;
  std::size_t negatives_per_edge;

  std::optional<std::size_t> val_start = std::nullopt;
  std::optional<std::size_t> test_start = std::nullopt;
};

class TGUFBuilder {
 public:
  explicit TGUFBuilder(const TGUFSchema& schema);
  ~TGUFBuilder();

  auto append_edges(const Batch& batch) const -> void;
  auto append_labels(const torch::Tensor& n_id, const torch::Tensor& time,
                     const torch::Tensor& target) const -> void;
  auto finalize() -> void;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class TGStore {
 public:
  enum class NegStrategy {
    None,         // Node Prop or Inference
    Random,       // Link Prop Training (1:1 random negatives)
    PreComputed,  // Link Prop Eval (Uses pre-comptued negatives)
  };

  struct IndexRange {
    IndexRange() = default;
    IndexRange(std::size_t s, std::size_t e) : start_(s), end_(e) {
      if (end_ < start_) {
        throw std::out_of_range("Invalid range");
      }
    }
    [[nodiscard]] auto start() const -> std::size_t { return start_; }
    [[nodiscard]] auto end() const -> std::size_t { return end_; }
    [[nodiscard]] auto size() const -> std::size_t { return end_ - start_; }

    std::size_t start_{0};
    std::size_t end_{0};
  };

  virtual ~TGStore() = default;

  [[nodiscard]] static auto from_memory(
      const Batch& edges,
      const std::optional<torch::Tensor>& label_n_id = std::nullopt,
      const std::optional<torch::Tensor>& label_time = std::nullopt,
      const std::optional<torch::Tensor>& label_target = std::nullopt,
      std::optional<std::size_t> val_start = std::nullopt,
      std::optional<std::size_t> test_start = std::nullopt)
      -> std::shared_ptr<TGStore>;

  [[nodiscard]] static auto from_tguf(
      const std::string& path,
      std::optional<std::size_t> val_start = std::nullopt,
      std::optional<std::size_t> test_start = std::nullopt)
      -> std::shared_ptr<TGStore>;

  [[nodiscard]] virtual auto edge_count() const -> std::size_t = 0;
  [[nodiscard]] virtual auto node_count() const -> std::size_t = 0;
  [[nodiscard]] virtual auto msg_dim() const -> std::size_t = 0;
  [[nodiscard]] virtual auto label_dim() const -> std::size_t = 0;

  [[nodiscard]] virtual auto train_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto val_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto test_split() const -> IndexRange = 0;

  [[nodiscard]] virtual auto train_label_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto val_label_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto test_label_split() const -> IndexRange = 0;

  [[nodiscard]] virtual auto get_batch(
      std::size_t start, std::size_t size,
      NegStrategy strategy = NegStrategy::None) const -> Batch = 0;
  [[nodiscard]] virtual auto gather_timestamps(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;
  [[nodiscard]] virtual auto gather_msgs(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;

  [[nodiscard]] virtual auto get_edge_cutoff_for_label_event(
      std::size_t l_id) const -> std::size_t = 0;

  [[nodiscard]] virtual auto get_label_event(std::size_t l_id) const
      -> LabelEvent = 0;
};

class TGNImpl : public torch::nn::Module {
 public:
  TGNImpl(const TGNConfig& cfg, const std::shared_ptr<TGStore>& store);
  ~TGNImpl();

  auto detach_memory() -> void;
  auto reset_state() -> void;
  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& time, const torch::Tensor& msg)
      -> void;

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
