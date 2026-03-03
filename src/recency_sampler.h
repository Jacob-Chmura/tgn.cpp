#pragma once

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace tgn {

class LastNeighborLoader {
 public:
  LastNeighborLoader(std::size_t num_nbrs, std::size_t num_nodes);

  auto operator()(const torch::Tensor& global_n_id)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

  auto insert(const torch::Tensor& src, const torch::Tensor& dst) -> void;

  auto reset_state() -> void;

 private:
  std::int64_t buffer_size_{};
  std::int64_t cur_e_id_{0};

  torch::Tensor buffer_nbrs_;
  torch::Tensor buffer_e_id_;
  torch::Tensor assoc_;
};

}  // namespace tgn
