#include <cstdint>

namespace tguf {

/** @brief Magic number used to identify valid TGUF files.
 * Stored at the beginning of the file header for format validation.
 */
static constexpr std::uint64_t TGUF_MAGIC = 0x54474E42494E3031;

/** @brief Version of the TGUF file format.
 * Used for backward/forward compatibility when reading files.
 */
static constexpr std::uint64_t TGUF_VERSION = 1;

/** @brief Memory/page alignment size in bytes.
 * Ensures all sections in the TGUF file are page-aligned for efficient mmap
 * access.
 */
static constexpr std::size_t TGUF_PAGE = 4096;

/** @struct TGUFHeader
 * @brief Header metadata baked into the TGUF binary.
 */
struct alignas(8) TGUFHeader {
  std::uint64_t magic = TGUF_MAGIC;
  std::uint64_t version = TGUF_VERSION;

  std::uint64_t num_edges{};
  std::uint64_t num_labels{};
  std::uint64_t num_nodes{};
  std::uint64_t msg_dim{};
  std::uint64_t label_dim{};
  std::uint64_t node_feat_dim{};
  std::uint64_t negatives_start_e_id{};
  std::uint64_t negatives_per_edge{};

  std::uint64_t src_offset{};
  std::uint64_t dst_offset{};
  std::uint64_t time_offset{};
  std::uint64_t msg_offset{};
  std::uint64_t neg_dst_offset{};
  std::uint64_t label_n_id_offset{};
  std::uint64_t label_time_offset{};
  std::uint64_t label_target_offset{};
  std::uint64_t node_feat_offset{};

  std::uint64_t val_start{};
  std::uint64_t test_start{};
};

}  // namespace tguf
