#include "scatter_ops.h"

#include <torch/types.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace tgn {

auto inline prepare_scatter(const torch::Tensor& src,
                            const torch::Tensor& index, std::int64_t dim_size)
    -> std::pair<std::vector<std::int64_t>, torch::Tensor> {
  std::vector<std::int64_t> out_shape = {dim_size};
  auto idx = index;
  if (src.dim() == 2) {
    out_shape.push_back(src.size(1));
    idx = index.unsqueeze(-1).expand_as(src);
  }
  return {out_shape, idx};
}

auto scatter_max(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor {
  const auto [shape, idx] = prepare_scatter(src, index, dim_size);
  auto out = torch::zeros(shape, src.options());
  out.scatter_reduce_(0, idx, src, "amax", false);
  return out;
}

auto scatter_add(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor {
  const auto [shape, idx] = prepare_scatter(src, index, dim_size);
  auto out = torch::zeros(shape, src.options());
  out.scatter_add_(0, idx, src);
  return out;
}

auto scatter_softmax(const torch::Tensor& src, const torch::Tensor& index,
                     std::int64_t dim_size) -> torch::Tensor {
  const auto src_max = scatter_max(src.detach(), index, dim_size);
  auto out = src - src_max.index_select(0, index);
  out = out.exp();

  auto out_sum = scatter_add(out, index, dim_size) + 1e-16;
  out_sum = out_sum.index_select(0, index);
  return out / out_sum;
}

auto scatter_argmax(const torch::Tensor& src, const torch::Tensor& index,
                    std::int64_t dim_size) -> torch::Tensor {
  auto res = scatter_max(src, index, dim_size);
  auto out = torch::full({dim_size}, /*fill_value*/ dim_size - 1,
                         src.options().dtype(torch::kLong));

  // Find where edge values match the winning max for each node
  const auto mask = src == res.index_select(0, index);
  const auto nonzero = torch::nonzero(mask).view(-1);
  const auto target_indices = index.index_select(0, nonzero);
  out.index_put_({target_indices}, nonzero);
  return out;
}

}  // namespace tgn
