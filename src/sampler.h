#pragma once

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace tgn {

/** @class LastNeighborLoader
 * @brief A high-performance vectorized recency sampler.
 * This sampler maintains a persistent state of the $K$ most recent temporal
 * interactions for every node. It uses a "Top-K" merging strategy during
 * insertion to ensure that even within a single batch, only the most recent
 * unique events are preserved.
 */
class LastNeighborLoader {
 public:
  /**
   * @brief Constructs the sampler and allocates persistent buffers.
   * @param num_nbrs The number of most neighbors ($K$) to track per node.
   * @param num_nodes The total capacity of the node index space ($N$).
   */
  LastNeighborLoader(std::size_t num_nbrs, std::size_t num_nodes);

  /**
   * @brief Samples the temporal neighborhood and performs local relabeling.
   * * Retrieves the $K$ most recent neighbors for the requested batch and
   * transforms the global node IDs into a local coordinates $[0,
   * \text{unique\_nodes})$.
   * @param global_n_id Tensor of query node IDs $[B]$.
   * @return A tuple containing:
   * - **unique_n_id**: The set of all unique nodes in the subgraph $[U]$.
   * - **edge_index**: Relabeled adjacency matrix $[2, \text{valid\_edges}]$.
   * - **edge_ids**: The original global indices for the sampled edges.
   */
  auto operator()(const torch::Tensor& global_n_id)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

  /**
   * @brief Inserts new edge events using a Top-K temporal merge.
   * Processes the batch bi-directionally (src $\to$ dst and dst $\to$ src).
   * For each node, it merges history with new events and uses `topk`
   * to truncate back to $K$ entries based on the highest global edge IDs.
   * @param src Source node IDs $[B]$.
   * @param dst Destination node IDs $[B]$.
   */
  auto insert(const torch::Tensor& src, const torch::Tensor& dst) -> void;

  /** @brief Resets the edge counter and fills the ID buffer with -1. */
  auto reset_state() -> void;

 private:
  std::int64_t buffer_size_{};
  std::int64_t cur_e_id_{0};

  torch::Tensor buffer_nbrs_;
  torch::Tensor buffer_e_id_;
  torch::Tensor assoc_;
};

}  // namespace tgn
