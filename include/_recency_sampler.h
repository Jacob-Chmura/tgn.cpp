#include <cstdint>
#include <tuple>

namespace tgn::detail {

struct LastNeighborLoader {
  LastNeighborLoader(std::size_t num_nbrs, std::size_t num_nodes)
      : buffer_size_(static_cast<std::int64_t>(num_nbrs)),
        buffer_nbrs_(torch::empty({static_cast<std::int64_t>(num_nodes),
                                   static_cast<std::int64_t>(num_nbrs)},
                                  torch::TensorOptions().dtype(torch::kLong))),
        buffer_e_id_(torch::empty({static_cast<std::int64_t>(num_nodes),
                                   static_cast<std::int64_t>(num_nbrs)},
                                  torch::TensorOptions().dtype(torch::kLong))),
        assoc_(torch::empty({static_cast<std::int64_t>(num_nodes)},
                            torch::TensorOptions().dtype(torch::kLong))) {
    reset_state();
  }

  auto operator()(const torch::Tensor& global_n_id)
      -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    // Shape: [batch_size, sampler_size]
    const auto nbrs = buffer_nbrs_.index_select(0, global_n_id);
    const auto e_id = buffer_e_id_.index_select(0, global_n_id);

    // Shape: [batch_size, sampler_size]
    const auto nodes = global_n_id.view({-1, 1}).expand_as(nbrs);

    // Filter invalid neighbors (e_id < 0)
    const auto mask = e_id >= 0;
    const auto filtered_nbrs = nbrs.index({mask});
    const auto filtered_e_id = e_id.index({mask});
    const auto filtered_nodes = nodes.index({mask});

    // Relabel node indices and combine nodes with sampled neighbors
    const auto [unique_n_id, _] =
        at::_unique(torch::cat({global_n_id, filtered_nbrs}));
    assoc_.index_put_({unique_n_id}, torch::arange(unique_n_id.size(0),
                                                   unique_n_id.options()));

    if (filtered_nbrs.numel() == 0) {
      return std::make_tuple(unique_n_id,
                             torch::empty({2, 0}, unique_n_id.options()),
                             filtered_e_id);
    }

    // Map global IDs to local IDs [0, len(n_id) - 1]
    const auto local_nbrs = assoc_.index_select(0, filtered_nbrs);
    const auto local_nodes = assoc_.index_select(0, filtered_nodes);
    const auto edge_index = torch::stack({local_nbrs, local_nodes}, 0);
    return std::make_tuple(unique_n_id, edge_index, filtered_e_id);
  }

  auto insert(const torch::Tensor& src, const torch::Tensor& dst) -> void {
    // Collect central nodes, their nbrs and the current event ids.
    auto nbrs = torch::cat({src, dst}, 0);
    auto nodes = torch::cat({dst, src}, 0);

    // Create edge IDs for this batch and repeat for bi-directional edges
    const auto batch_size = src.size(0);
    auto e_id = torch::arange(cur_e_id_, cur_e_id_ + batch_size, src.options())
                    .repeat({2});
    cur_e_id_ += batch_size;

    // Sort interactions by node ID to simplify batch processing
    const auto [sort_out, perm] = nodes.sort();
    nodes = sort_out;
    nbrs = nbrs.index_select(0, perm);
    e_id = e_id.index_select(0, perm);

    // Find unique nodes and map to local range [0, num_unique - 1]
    auto [unique_out, _] = at::_unique(nodes);
    const auto n_id = unique_out;
    assoc_.index_put_({n_id}, torch::arange(n_id.size(0), n_id.options()));

    // Create "dense" temporary representation
    // dense_id determines the column in the [num_unique, size] window
    auto dense_id =
        torch::arange(nodes.size(0), nodes.options()) % buffer_size_;
    dense_id += assoc_.index_select(0, nodes).mul_(buffer_size_);

    const auto total_temp_slots = n_id.size(0) * buffer_size_;

    auto dense_e_id = torch::full({total_temp_slots}, -1, e_id.options());
    dense_e_id.index_put_({dense_id}, e_id);
    dense_e_id = dense_e_id.view({-1, buffer_size_});

    auto dense_nbrs = torch::empty({total_temp_slots}, nbrs.options());
    dense_nbrs.index_put_({dense_id}, nbrs);
    dense_nbrs = dense_nbrs.view({-1, buffer_size_});

    // Merge new interactions with existing ones in the global buffers
    // Fetch old data for the relevant nodes: shape [num_unique, size]
    const auto old_e_id = buffer_e_id_.index_select(0, n_id);
    const auto old_nbrs = buffer_nbrs_.index_select(0, n_id);

    // Concatenate old and new: shape [num_unique, size * 2]
    const auto merged_e_id = torch::cat({old_e_id, dense_e_id}, -1);
    const auto merged_nbrs = torch::cat({old_nbrs, dense_nbrs}, -1);

    // Keep only the 'size' most recent interactions (highest e_id)
    const auto [new_e_id, topk_perm] = merged_e_id.topk(buffer_size_, -1);

    // Use gather to pick the corresponding neighbors
    const auto new_nbrs = torch::gather(merged_nbrs, 1, topk_perm);

    // Write back to global buffers
    buffer_e_id_.index_put_({n_id}, new_e_id);
    buffer_nbrs_.index_put_({n_id}, new_nbrs);
  }

  auto reset_state() -> void {
    cur_e_id_ = 0;
    buffer_e_id_.fill_(-1);
  }

  std::int64_t buffer_size_{};
  std::int64_t cur_e_id_{0};

  torch::Tensor buffer_nbrs_;
  torch::Tensor buffer_e_id_;
  torch::Tensor assoc_;
};

}  // namespace tgn::detail
