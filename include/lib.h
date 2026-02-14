#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

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
  virtual auto size() const -> std::size_t = 0;

  [[nodiscard]] virtual auto get_batch(std::size_t start,
                                       std::size_t batch_size) const
      -> Batch = 0;

  [[nodiscard]] virtual auto fetch_t(const torch::Tensor e_id) const
      -> torch::Tensor = 0;
  [[nodiscard]] virtual auto fetch_msg(const torch::Tensor e_id) const
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

  auto detach_memory() -> void;
  auto reset_state() -> void;
  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& t, const torch::Tensor& msg) -> void;

  template <typename... Ts>
  auto forward(const Ts&... inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

TORCH_MODULE(TGN);

}  // namespace tgn
