#pragma once

#include <torch/types.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

/** @namespace tguf
 * @brief Temporal Graph Unified Format: A Temporal Graph Stream Format.
 */
namespace tguf {

/** @struct Batch
 * @brief Container for temporal edge data.
 */
struct Batch {
  torch::Tensor src;   ///< Source node IDs [B]
  torch::Tensor dst;   ///< Destination node IDs [B]
  torch::Tensor time;  ///< Timestamps [B]
  torch::Tensor msg;   ///< Edge features [B, msg_dim]
  std::optional<torch::Tensor>
      neg_dst;  ///< Optional negative destinations for link prediction [B,
                ///< negatives_per_edge]
};

/** @struct LabelEvent
 * @brief Container for a label event at a single point in time.
 */
struct LabelEvent {
  torch::Tensor n_id;    ///< Label Node Ids [B]
  torch::Tensor target;  ///< Label targets [B, label_dim]
};

/** @struct TGUFSchema
 * @brief metadata defining the layout of a Temporal Graph Unified Format file.
 */
struct TGUFSchema {
  std::string path;  ///< Path to .tguf binary.

  std::size_t edge_capacity;       ///< Max number of edges.
  std::size_t label_capacity;      ///< Max number of label events.
  std::size_t node_feat_capacity;  ///< Max nodes with static features.
  std::size_t msg_dim;             ///< Fixed edge feature dimension.
  std::size_t label_dim;           ///< Fixed label target dimension.
  std::size_t node_feat_dim;       ///< Fixed static node feature dimension.

  /** @brief For link prediction evaluation, the e_id where pre-computed
   * negatives begin. */
  std::size_t negatives_start_e_id;
  std::size_t negatives_per_edge;  ///< Fixed number of negatives per edge.

  /** * @brief Global index offset where the validation split begins.
   * If `std::nullopt`, the dataset is treated as 100% training data unless
   * overridden during @ref TGStore initialization.
   */
  std::optional<std::size_t> val_start = std::nullopt;

  /** * @brief Global index offset where the test split begins.
   * Must be greater than or equal to @ref val_start if both are provided.
   */
  std::optional<std::size_t> test_start = std::nullopt;
};

/** @class TGUFBuilder
 * @brief High-performance writer for creating TGUF datasets on disk.
 * Uses an internal buffer strategy to minimize disk I/O.
 */
class TGUFBuilder {
 public:
  explicit TGUFBuilder(const TGUFSchema& schema);
  ~TGUFBuilder();

  /** @brief Appends a batch of edges to the persistent store. */
  auto append_edges(const Batch& batch) const -> void;

  /** @brief Appends a batch of label events to the persistent store. */
  auto append_labels(const torch::Tensor& n_id, const torch::Tensor& time,
                     const torch::Tensor& target) const -> void;

  /** @brief Appends a batch of static node features to the persistent store. */
  auto append_node_feats(const torch::Tensor& n_id,
                         const torch::Tensor& node_feat) const -> void;

  /** @brief Finalizes the .tguf file, writing headers and flushing buffers. */
  auto finalize() -> void;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** @class TGStore
 * @brief Abstract interface for temporal graph storage.
 * Implementations can be purely in-memory or memory-mapped TGUF files.
 */
class TGStore {
 public:
  /** @enum NegStrategy
   * @brief Determines how negative samples are generated during get_batch().
   */
  enum class NegStrategy {
    None,         ///< No negatives (inference or node-level tasks).
    Random,       ///< Samples one random negative node per edge.
    PreComputed,  ///< Uses the fixed negatives stored in TGUF (for  eval).
  };

  /** @brief A contiguous slice of the graph (e.g., training split).
   */
  struct IndexRange {
    IndexRange() = default;
    IndexRange(std::size_t s, std::size_t e) : start_(s), end_(e) {
      if (end_ < start_) {
        throw std::out_of_range("Invalid range");
      }
    }
    [[nodiscard]] auto start() const -> std::size_t { return start_; }
    [[nodiscard]] auto end() const -> std::size_t { return end_; }
    [[nodiscard]] auto size() const -> std::size_t { return end_ - start_; }

    std::size_t start_{0};
    std::size_t end_{0};
  };

  virtual ~TGStore() = default;

  /** @brief Factory method for a high-speed, purely RAM-based store. */
  [[nodiscard]] static auto from_memory(
      const Batch& edges,
      const std::optional<torch::Tensor>& node_feats = std::nullopt,
      const std::optional<torch::Tensor>& label_n_id = std::nullopt,
      const std::optional<torch::Tensor>& label_time = std::nullopt,
      const std::optional<torch::Tensor>& label_target = std::nullopt,
      std::optional<std::size_t> val_start = std::nullopt,
      std::optional<std::size_t> test_start = std::nullopt)
      -> std::unique_ptr<TGStore>;

  /** @brief Factory method for memory-mapped storage from a TGUF file.
   * Supports datasets larger than available system RAM.
   */
  [[nodiscard]] static auto from_tguf(
      const std::string& path,
      std::optional<std::size_t> val_start = std::nullopt,
      std::optional<std::size_t> test_start = std::nullopt)
      -> std::unique_ptr<TGStore>;

  [[nodiscard]] virtual auto edge_count() const -> std::size_t = 0;
  [[nodiscard]] virtual auto node_count() const -> std::size_t = 0;
  [[nodiscard]] virtual auto label_count() const -> std::size_t = 0;
  [[nodiscard]] virtual auto msg_dim() const -> std::size_t = 0;
  [[nodiscard]] virtual auto label_dim() const -> std::size_t = 0;
  [[nodiscard]] virtual auto node_feat_dim() const -> std::size_t = 0;

  [[nodiscard]] virtual auto train_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto val_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto test_split() const -> IndexRange = 0;

  [[nodiscard]] virtual auto train_label_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto val_label_split() const -> IndexRange = 0;
  [[nodiscard]] virtual auto test_label_split() const -> IndexRange = 0;

  /** * @brief Retrieves a zero-copy slice of the graph.
   * @param start The starting edge ID.
   * @param size The number of edges to include.
   * @param strategy The negative sampling strategy to apply.
   * @param device The torch::Device to materialize the batch on.
   */
  [[nodiscard]] virtual auto get_batch(
      std::size_t start, std::size_t size,
      NegStrategy strategy = NegStrategy::None, torch::Device device = torch::kCPU) const -> Batch = 0;

  /** * @brief Performs a vectorized random-access gather of edge timestamps.
   * @param e_id Tensor of edge indices [num_edges].
   * @return torch::Tensor of timestamps [num_edges].
   * @note Optimized for memory-mapped I/O; performance may vary based on disk
   * locality.
   */
  [[nodiscard]] virtual auto gather_timestamps(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;

  /** * @brief Performs a vectorized random-access gather of edge messages.
   * @param e_id Tensor of edge indices [num_edges].
   * @return torch::Tensor of messages [num_edges, msg_dim].
   */
  [[nodiscard]] virtual auto gather_msgs(const torch::Tensor& e_id) const
      -> torch::Tensor = 0;

  /** * @brief Performs a vectorized random-access gather of node features.
   * @param n_id Tensor of node indices [num_nodes].
   * @return torch::Tensor of features [num_nodes, node_feat_dim].
   */
  [[nodiscard]] virtual auto gather_node_feats(const torch::Tensor& n_id) const
      -> torch::Tensor = 0;

  /** * @brief Retrieves the maximum edge_id that can be safely processed before
   * a label.
   * * To prevent information leakage (look-ahead bias), the model state should
   * only be updated with edges occurring before the timestamp of the label
   * event @p l_id.
   * * @param l_id The index of the label event.
   * @return The upper-bound edge_id (exclusive) for model state updates.
   */
  [[nodiscard]] virtual auto get_edge_cutoff_for_label_event(
      std::size_t l_id) const -> std::size_t = 0;

  /** * @brief Retrieves the metadata and target for a specific label event.
   * @param l_id The index of the label event.
   * @param device The torch::Device to materialize data on.
   * @return A @ref LabelEvent containing affected node IDs and target values.
   */
  [[nodiscard]] virtual auto get_label_event(std::size_t l_id, torch::Device device = torch::kCPU) const
      -> LabelEvent = 0;
};

}  // namespace tguf
