#include <torch/types.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "tgn.h"

namespace {
struct StreamHeader {
  std::uint64_t n_edges{};
  std::uint64_t m_dim{};
  std::uint64_t n_neg{};
  std::uint64_t n_labels{};
  std::uint64_t l_dim{};
  std::uint64_t val_start{};
  std::uint64_t test_start{};
};

auto read_exactly(void* ptr, std::size_t n_bytes) -> void {
  if (!std::cin.read(static_cast<char*>(ptr), n_bytes)) {
    throw std::runtime_error("Failed to read " + std::to_string(n_bytes) +
                             " bytes from stdin");
  }
}

struct Scratch {
  std::vector<std::int64_t> i64;
  std::vector<float> f32;

  auto reserve_edges(std::size_t bsize, std::size_t m_dim, std::size_t n_neg)
      -> void {
    i64.assign(bsize * (3 + n_neg), 0);  // src,dst,t + neg_dst
    f32.assign(bsize * m_dim, 0.0);
  }

  auto reserve_labels(std::size_t bsize, std::size_t l_dim) -> void {
    i64.assign(bsize * 2, 0);  // n_id, t
    f32.assign(bsize * l_dim, 0.0);
  }
};

auto process_edge_batch(const StreamHeader& h, const tgn::TGUFBuilder& builder,
                        std::int64_t bsize, Scratch& scratch) -> void {
  scratch.reserve_edges(bsize, h.m_dim, h.n_neg);
  read_exactly(scratch.i64.data(), bsize * 3 * sizeof(std::int64_t));
  read_exactly(scratch.f32.data(), bsize * h.m_dim * sizeof(float));
  if (h.n_neg > 0) {
    read_exactly(scratch.i64.data() + (bsize * 3),
                 bsize * h.n_neg * sizeof(std::int64_t));
  }
  tgn::Batch batch{
      .src =
          torch::from_blob(scratch.i64.data(), {bsize}, torch::kInt64).clone(),
      .dst =
          torch::from_blob(scratch.i64.data() + bsize, {bsize}, torch::kInt64)
              .clone(),
      .time = torch::from_blob(scratch.i64.data() + (bsize * 2), {bsize},
                               torch::kInt64)
                  .clone(),
      .msg = torch::from_blob(scratch.f32.data(),
                              {bsize, static_cast<std::int64_t>(h.m_dim)},
                              torch::kFloat32)
                 .clone()};

  if (h.n_neg > 0) {
    batch.neg_dst =
        torch::from_blob(scratch.i64.data() + (bsize * 3),
                         {bsize, static_cast<std::int64_t>(h.n_neg)},
                         torch::kInt64)
            .clone();
  }
  builder.append_edges(batch);
}

auto process_label_batch(const StreamHeader& h, const tgn::TGUFBuilder& builder,
                         std::int64_t bsize, Scratch& scratch) -> void {
  scratch.reserve_labels(bsize, h.l_dim);
  read_exactly(scratch.i64.data(), bsize * 2 * sizeof(std::int64_t));
  read_exactly(scratch.f32.data(), bsize * h.l_dim * sizeof(float));

  builder.append_labels(
      torch::from_blob(scratch.i64.data(), {bsize}, torch::kInt64).clone(),
      torch::from_blob(scratch.i64.data() + bsize, {bsize}, torch::kInt64)
          .clone(),
      torch::from_blob(scratch.f32.data(),
                       {bsize, static_cast<std::int64_t>(h.l_dim)},
                       torch::kFloat32)
          .clone());
}

}  // namespace

auto main(int argc, char** argv) -> int {
  if (argc < 2) {
    std::cerr << "Usage: tguf_cli <out_path>\n";
    return 1;
  }
  std::string out_path = argv[1];
  try {
    StreamHeader h;
    read_exactly(&h, sizeof(StreamHeader));

    tgn::TGUFSchema schema{
        .path = out_path,
        .edge_capacity = h.n_edges,
        .label_capacity = h.n_labels,
        .msg_dim = h.m_dim,
        .label_dim = h.l_dim,
        .num_negatives = h.n_neg,
        .val_start = h.val_start,
        .test_start = h.test_start,
    };

    tgn::TGUFBuilder builder(schema);
    Scratch scratch;
    char cmd;
    std::int64_t bsize;
    while (std::cin.read(&cmd, 1)) {
      read_exactly(&bsize, sizeof(std::int64_t));
      if (bsize <= 0) {
        continue;
      }

      if (cmd == 'E') {
        process_edge_batch(h, builder, bsize, scratch);
      } else if (cmd == 'L') {
        process_label_batch(h, builder, bsize, scratch);
      }
    }
    builder.finalize();
  } catch (const std::exception& e) {
    std::cerr << "Error in tguf_cli: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
