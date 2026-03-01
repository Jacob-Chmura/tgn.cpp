#pragma once
#include <torch/torch.h>

#include <cstdint>

#include "lib.h"

namespace tgn {
static constexpr std::uint64_t TGUF_MAGIC = 0x54474E42494E3031;
static constexpr std::uint64_t TGUF_VERSION = 1;

struct alignas(8) TGUFHeader {
  std::uint64_t magic = TGUF_MAGIC;
  std::uint64_t version = TGUF_VERSION;

  std::uint64_t num_edges = 0;
  std::uint64_t msg_dim = 0;
  std::uint64_t n_neg = 0;
  std::uint64_t num_labels = 0;
  std::uint64_t label_dim = 0;

  std::uint64_t src_offset = 0;
  std::uint64_t dst_offset = 0;
  std::uint64_t t_offset = 0;
  std::uint64_t msg_offset = 0;
  std::uint64_t neg_dst_offset = 0;
  std::uint64_t label_n_id_offset = 0;
  std::uint64_t label_t_offset = 0;
  std::uint64_t label_y_true_offset = 0;
};

}  // namespace tgn
