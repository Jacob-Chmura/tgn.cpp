#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tguf.h"

/** @namespace tgn
 * @brief Temporal Graph Networks: A High-performance implementation.
 */
namespace tgn {

/** @struct TGNConfig
 * @brief Configuration parameters for the TGN model architecture.
 */
struct TGNConfig {
  std::size_t embedding_dim = 100;  ///< TransformerConv embedding size.
  std::size_t memory_dim = 100;     ///< TGNMemory embedding size.
  std::size_t time_dim = 100;       ///< TimeEncoder embedding size.
  std::size_t num_heads = 2;  ///< TransformerConv multi-head attention heads.
  std::size_t num_nbrs = 10;  ///< RecencySampler neighbor buffer size.
  float dropout = 0.1;        ///< TransformerConv dropout.
};

/** @class TGNImpl
 * @brief The core Temporal Graph Network module.
 * Manages node memory state and temporal neighborhood aggregation.
 */
class TGNImpl : public torch::nn::Module {
 public:
  TGNImpl(const TGNConfig& cfg, const std::shared_ptr<tguf::TGStore>& store);
  ~TGNImpl();

  /** @brief Detaches memory from the computational graph to truncate backprop
   * (BPTT). */
  auto detach_memory() -> void;

  /** @brief Zeros out all node memory and resets last-update timestamps. */
  auto reset_state() -> void;

  /** @brief Updates internal memory given a batch of true edge events. */
  auto update_state(const torch::Tensor& src, const torch::Tensor& dst,
                    const torch::Tensor& time, const torch::Tensor& msg)
      -> void;

  /**
   * @brief Variadic forward pass.
   * @param inputs Tensors of node IDs to compute embeddings for.
   * @return A tuple of embeddings [B, embedding_dim] in same order as inputs.
   */
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
