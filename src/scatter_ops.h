#pragma once
#include <torch/types.h>

#include <cstdint>

namespace tgn {

auto scatter_max(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor;

auto scatter_add(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor;

auto scatter_softmax(const torch::Tensor& src, const torch::Tensor& index,
                     std::int64_t dim_size) -> torch::Tensor;

auto scatter_argmax(const torch::Tensor& src, const torch::Tensor& index,
                    std::int64_t dim_size) -> torch::Tensor;
}  // namespace tgn
