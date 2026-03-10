#pragma once

#include <cstddef>
#include <cstdint>

namespace tgn {
static constexpr std::uint64_t TGUF_MAGIC = 0x54474E42494E3031;
static constexpr std::uint64_t TGUF_VERSION = 1;
static constexpr std::size_t TGUF_PAGE = 4096;

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

}  // namespace tgn
