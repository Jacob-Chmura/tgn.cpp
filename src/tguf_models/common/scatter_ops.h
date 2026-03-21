#pragma once
#include <torch/types.h>

#include <cstdint>

namespace tgn {

/** * @brief Vectorized scatter-max reduction.
 * Reduces values from @p src into segments defined by @p index.
 * @note If a segment has no entries, the output is initialized to 0.0.
 */
auto scatter_max(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor;

/** * @brief Vectorized scatter-add reduction.
 * Accumulates values from @p src into segments defined by @p index.
 * @note Empty segments result in 0.0.
 */
auto scatter_add(const torch::Tensor& src, const torch::Tensor& index,
                 std::int64_t dim_size) -> torch::Tensor;

/** * @brief Segmented softmax over irregular tensor groups.
 * Computes a numerically stable softmax for each segment in @p src.
 * @note Stability is achieved via subtract-max. A small epsilon (1e-16)
 * is added to the denominator to prevent NaN for nodes with zero neighbors.
 */
auto scatter_softmax(const torch::Tensor& src, const torch::Tensor& index,
                     std::int64_t dim_size) -> torch::Tensor;

/** * @brief Vectorized scatter-argmax.
 * Identifies the index in @p src that corresponds to the maximum value
 * within each segment.
 * @note If multiple indices hold the same max value, last one is selected.
 */
auto scatter_argmax(const torch::Tensor& src, const torch::Tensor& index,
                    std::int64_t dim_size) -> torch::Tensor;

}  // namespace tgn
