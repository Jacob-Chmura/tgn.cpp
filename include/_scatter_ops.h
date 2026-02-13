#include <cstdint>

namespace tgn::detail {

auto scatter_max(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor {
  return src.new_zeros(dim_size).scatter_reduce_(/*dim*/ 0, index, src,
                                                 /* reduce */ "amax",
                                                 /* include_self*/ false);
}

auto scatter_add(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor {
  return src.new_zeros(dim_size).scatter_reduce_(/*dim*/ 0, index, src,
                                                 /* reduce */ "sum",
                                                 /* include_self*/ false);
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
  auto out = index.new_full({dim_size}, /*fill_value*/ dim_size - 1);

  // Find where edge values match the winning max for each node
  const auto mask = src == res.index_select(0, index);
  const auto nonzero = torch::nonzero(mask).view(-1);
  const auto target_indices = index.index_select(0, nonzero);
  out.index_put_({target_indices}, nonzero);
  return out;
}

}  // namespace tgn::detail
