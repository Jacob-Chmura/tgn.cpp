#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
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
  torch::Tensor t;
  torch::Tensor msg;
  torch::Tensor neg_dst;  // TODO(kuba): use std::optional<>
};

class TGStore {
 public:
  virtual ~TGStore() = default;

  [[nodiscard]] virtual auto num_edges() const -> std::size_t = 0;
  [[nodiscard]] virtual auto num_nodes() const -> std::size_t = 0;
  [[nodiscard]] virtual auto msg_dim() const -> std::size_t = 0;

  [[nodiscard]] virtual auto get_batch(std::size_t start,
                                       std::size_t size) const -> Batch = 0;
  [[nodiscard]] virtual auto get_t(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;
  [[nodiscard]] virtual auto get_msg(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;
};

struct DummyTGStoreOptions {};

struct InMemoryTGStoreOptions {
  torch::Tensor src;
  torch::Tensor dst;
  torch::Tensor t;
  torch::Tensor msg;
  torch::Tensor neg_dst;
};

auto make_store(const DummyTGStoreOptions& opts) -> std::unique_ptr<TGStore>;
auto make_store(const InMemoryTGStoreOptions& opts) -> std::unique_ptr<TGStore>;

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
