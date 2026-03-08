#include "tguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <torch/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include "logging.h"
#include "tgn.h"

namespace tgn {

struct TGUFBuilder::Impl {
  TGUFSchema schema{};
  TGUFHeader header{};

  void* base_ptr = nullptr;
  bool finalized = false;
  std::size_t written_edges{};
  std::size_t written_labels{};
  std::size_t mapped_bytes{};

  explicit Impl(const TGUFSchema& schema) : schema(schema) {
    TGN_LOG_INFO("TGUFBuilder: Creating TGUF binary at {}", schema.path);
    header.msg_dim = schema.msg_dim;
    header.label_dim = schema.label_dim;
    header.negatives_start_e_id = schema.negatives_start_e_id;
    header.negatives_per_edge = schema.negatives_per_edge;
    header.val_start = schema.val_start.value_or(0);
    header.test_start = schema.test_start.value_or(0);

    auto align = [](std::size_t size) {
      return (size + TGUF_PAGE - 1) & ~(TGUF_PAGE - 1);
    };

    header.src_offset = TGUF_PAGE;  // Header gets its own page
    header.dst_offset =
        header.src_offset + align(schema.edge_capacity * sizeof(std::int64_t));
    header.time_offset =
        header.dst_offset + align(schema.edge_capacity * sizeof(std::int64_t));
    header.msg_offset =
        header.time_offset + align(schema.edge_capacity * sizeof(std::int64_t));

    std::size_t last_offset =
        header.msg_offset +
        align(schema.edge_capacity * schema.msg_dim * sizeof(float));

    if (schema.negatives_per_edge > 0) {
      header.neg_dst_offset = last_offset;
      last_offset +=
          align((schema.edge_capacity - header.negatives_start_e_id) *
                schema.negatives_per_edge * sizeof(std::int64_t));
    }

    if (schema.label_capacity > 0) {
      header.label_n_id_offset = last_offset;
      header.label_time_offset =
          header.label_n_id_offset +
          align(schema.label_capacity * sizeof(std::int64_t));
      header.label_target_offset =
          header.label_time_offset +
          align(schema.label_capacity * sizeof(std::int64_t));
      last_offset =
          header.label_target_offset +
          align(schema.label_capacity * schema.label_dim * sizeof(float));
    }

    mapped_bytes = last_offset;

    TGN_LOG_INFO(
        "TGUFBuilder: Pre-allocating {:.2f} GiB for {} edges and {} labels "
        "(msg_dim={}, label_dim={}, negatives_per_edge={})",
        mapped_bytes / (1024.0 * 1024.0 * 1024.0), schema.edge_capacity,
        schema.label_capacity, header.msg_dim, header.label_dim,
        header.negatives_per_edge);

    if (header.val_start > 0 || header.test_start > 0) {
      TGN_LOG_INFO(
          "TGUFBuilder: Using hardcoded edge splits (Val Start: {}, Test "
          "Start: {})",
          header.val_start, header.test_start);
    }

    auto fd = open(schema.path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) {
      throw std::runtime_error("Failed to create file");
    }

#ifdef __APPLE__
    fstore_t store = {};
    store.fst_flags = F_ALLOCATECONTIG;  // try contiguous first
    store.fst_posmode = F_PEOFPOSMODE;
    store.fst_offset = 0;
    store.fst_length = mapped_bytes;
    store.fst_bytesalloc = 0;

    if (fcntl(fd, F_PREALLOCATE, &store) == -1) {
      // Fall back to non-contiguous allocation
      store.fst_flags = F_ALLOCATEALL;
      if (fcntl(fd, F_PREALLOCATE, &store) == -1) {
        close(fd);
        throw std::runtime_error("Failed to preallocate disk space (macOS)");
      }
    }

    // Set the logical file size
    if (ftruncate(fd, mapped_bytes) != 0) {
      close(fd);
      throw std::runtime_error("Failed to set file size (macOS)");
    }
#else
    if (posix_fallocate(fd, 0, mapped_bytes) != 0) {
      close(fd);
      throw std::runtime_error("Failed to allocate disk space (Linux)");
    }
#endif

    base_ptr =
        mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base_ptr == MAP_FAILED) {
      throw std::runtime_error("Mmap for builder failed");
    }
  }

  ~Impl() {
    if (!finalized) {
      munmap(base_ptr, mapped_bytes);
    }
  }

  auto to_mmap(std::uint64_t offset, std::size_t start_idx,
               std::size_t element_size, const torch::Tensor& t) const -> void {
    auto t_contiguous = t.contiguous();
    TORCH_CHECK(t_contiguous.nbytes() == t.size(0) * element_size,
                "Tensor size mismatch with expected element size");
    auto* dst = static_cast<std::uint8_t*>(base_ptr) + offset +
                (start_idx * element_size);
    std::memcpy(dst, t_contiguous.data_ptr(), t_contiguous.nbytes());
  }
};

TGUFBuilder::TGUFBuilder(const TGUFSchema& schema)
    : impl_(std::make_unique<Impl>(schema)) {}
TGUFBuilder::~TGUFBuilder() = default;

auto TGUFBuilder::append_edges(const Batch& batch) const -> void {
  if (impl_->finalized) {
    throw std::runtime_error(
        "TGUFBuilder::append_edges: Cannot append to a finalized file.");
  }

  auto count = batch.src.size(0);
  if (count == 0) {
    return;
  }
  TGN_LOG_DEBUG("TGUFBuilder: Appending {} edges to TGUF file", count);

  if (impl_->written_edges + count > impl_->schema.edge_capacity) {
    throw std::runtime_error(
        "TGUFBuilder::append_edges: Overflow. Attempting to write " +
        std::to_string(count) + " edges, but only " +
        std::to_string(impl_->schema.edge_capacity - impl_->written_edges) +
        " slots remain.");
  }
  if (batch.msg.size(1) != static_cast<std::int64_t>(impl_->header.msg_dim)) {
    throw std::invalid_argument(
        "TGUFBuilder::append_edges: Message dimension mismatch. Expected " +
        std::to_string(impl_->header.msg_dim) + ", got " +
        std::to_string(batch.msg.size(1)));
  }
  if (impl_->header.negatives_per_edge > 0) {
    if (!batch.neg_dst.has_value()) {
      throw std::invalid_argument(
          "TGUFBuilder::append_edges: neg_dst is required for this TGUF "
          "schema.");
    }
    if (batch.neg_dst->size(1) !=
        static_cast<std::int64_t>(impl_->header.negatives_per_edge)) {
      throw std::invalid_argument(
          "TGUFBuilder::append_edges: Negative destination dimension mismatch. "
          "Expected " +
          std::to_string(impl_->header.negatives_per_edge) + ", got " +
          std::to_string(batch.neg_dst->size(1)));
    }
  }

  impl_->to_mmap(impl_->header.src_offset, impl_->written_edges,
                 sizeof(std::int64_t), batch.src);
  impl_->to_mmap(impl_->header.dst_offset, impl_->written_edges,
                 sizeof(std::int64_t), batch.dst);
  impl_->to_mmap(impl_->header.time_offset, impl_->written_edges,
                 sizeof(std::int64_t), batch.time);
  impl_->to_mmap(impl_->header.msg_offset, impl_->written_edges,
                 impl_->header.msg_dim * sizeof(float), batch.msg);

  if (batch.neg_dst.has_value() && impl_->header.neg_dst_offset > 0) {
    const auto batch_start = impl_->written_edges;
    const auto batch_end = batch_start + count;
    const auto neg_start = impl_->header.negatives_start_e_id;

    if (batch_end > neg_start) {
      // Calculate which part of the batch is valid negatives
      const auto slice_start =
          std::max(0L, static_cast<std::int64_t>(neg_start - batch_start));
      const auto mmap_row_idx =
          std::max(0L, static_cast<std::int64_t>(batch_start - neg_start));
      const auto valid_slice = batch.neg_dst->slice(0, slice_start);
      impl_->to_mmap(impl_->header.neg_dst_offset, mmap_row_idx,
                     impl_->header.negatives_per_edge * sizeof(int64_t),
                     valid_slice);
    }
  }
  impl_->written_edges += count;
}

auto TGUFBuilder::append_labels(const torch::Tensor& n_id,
                                const torch::Tensor& time,
                                const torch::Tensor& target) const -> void {
  if (impl_->finalized) {
    throw std::runtime_error(
        "TGUFBuilder::append_labels: Cannot append labels to a finalized "
        "file.");
  }

  const auto count = n_id.size(0);
  if (count == 0) {
    return;
  }
  TGN_LOG_DEBUG("TGUFBuilder: Appending {} labels to TGUF file", count);

  if (impl_->written_labels + count > impl_->schema.label_capacity) {
    throw std::runtime_error(
        "TGUFBuilder::append_labels: Overflow. Writing " +
        std::to_string(count) + " labels, but only " +
        std::to_string(impl_->schema.label_capacity - impl_->written_labels) +
        " slots remain.");
  }

  if (target.size(1) != static_cast<std::int64_t>(impl_->header.label_dim)) {
    throw std::invalid_argument(
        "TGUFBuilder::append_labels: Label dimension mismatch. Expected " +
        std::to_string(impl_->header.label_dim) + ", got " +
        std::to_string(target.size(1)));
  }

  impl_->to_mmap(impl_->header.label_n_id_offset, impl_->written_labels,
                 sizeof(std::int64_t), n_id);
  impl_->to_mmap(impl_->header.label_time_offset, impl_->written_labels,
                 sizeof(std::int64_t), time);
  impl_->to_mmap(impl_->header.label_target_offset, impl_->written_labels,
                 impl_->header.label_dim * sizeof(float), target);
  impl_->written_labels += count;
}

auto TGUFBuilder::finalize() -> void {
  if (impl_->finalized) {
    return;
  }

  if (impl_->written_edges < impl_->schema.edge_capacity) {
    TGN_LOG_WARN(
        "TGUFBuilder: Finalizing with fewer edges than declared ({} < {}). "
        "File will have unused padding.",
        impl_->written_edges, impl_->schema.edge_capacity);
  }

  // Update header with counts (user might have wrote less than declared)
  impl_->header.num_edges = impl_->written_edges;
  impl_->header.num_labels = impl_->written_labels;
  std::memcpy(impl_->base_ptr, &impl_->header, sizeof(TGUFHeader));

  msync(impl_->base_ptr, impl_->mapped_bytes, MS_SYNC);
  munmap(impl_->base_ptr, impl_->mapped_bytes);
  impl_->base_ptr = nullptr;
  impl_->finalized = true;

  TGN_LOG_INFO(
      "TGUFBuilder: Finalized to {} (Total edges: {}, Total labels: {})",
      impl_->schema.path, impl_->header.num_edges, impl_->header.num_labels);
}
}  // namespace tgn
