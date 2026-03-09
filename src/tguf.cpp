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
  TGUFSchema schema_{};
  TGUFHeader header_{};

  void* base_ptr_ = nullptr;
  bool finalized_ = false;
  std::size_t written_edges_{};
  std::size_t written_labels_{};
  std::size_t written_negatives_{};
  std::size_t mapped_bytes_{};

  explicit Impl(const TGUFSchema& schema) : schema_(schema) {
    TGN_LOG_INFO("TGUFBuilder: Creating TGUF binary at {}", schema.path);
    header_.msg_dim = schema.msg_dim;
    header_.label_dim = schema.label_dim;
    header_.negatives_per_edge = schema.negatives_per_edge;
    header_.val_start = schema.val_start.value_or(0);
    header_.test_start = schema.test_start.value_or(0);

    auto align = [](std::size_t size) {
      return (size + TGUF_PAGE - 1) & ~(TGUF_PAGE - 1);
    };

    header_.src_offset = TGUF_PAGE;  // Header gets its own page
    header_.dst_offset =
        header_.src_offset + align(schema.edge_capacity * sizeof(std::int64_t));
    header_.time_offset =
        header_.dst_offset + align(schema.edge_capacity * sizeof(std::int64_t));
    header_.msg_offset = header_.time_offset +
                         align(schema.edge_capacity * sizeof(std::int64_t));

    std::size_t last_offset =
        header_.msg_offset +
        align(schema.edge_capacity * schema.msg_dim * sizeof(float));

    if (schema.negatives_capacity > 0 && schema.negatives_per_edge > 0) {
      header_.neg_dst_offset = last_offset;
      last_offset += align(schema.negatives_capacity *
                           schema.negatives_per_edge * sizeof(std::int64_t));
    }

    if (schema.label_capacity > 0) {
      header_.label_n_id_offset = last_offset;
      header_.label_time_offset =
          header_.label_n_id_offset +
          align(schema.label_capacity * sizeof(std::int64_t));
      header_.label_target_offset =
          header_.label_time_offset +
          align(schema.label_capacity * sizeof(std::int64_t));
      last_offset =
          header_.label_target_offset +
          align(schema.label_capacity * schema.label_dim * sizeof(float));
    }

    mapped_bytes_ = last_offset;

    TGN_LOG_INFO(
        "TGUFBuilder: Pre-allocating {:.2f} GiB for {} edges and {} labels "
        "(msg_dim={}, label_dim={}, negatives_capacity={}, "
        "negatives_per_edge={})",
        mapped_bytes_ / (1024.0 * 1024.0 * 1024.0), schema.edge_capacity,
        schema.label_capacity, header_.msg_dim, header_.label_dim,
        schema.negatives_capacity, header_.negatives_per_edge);

    if (header_.val_start > 0 || header_.test_start > 0) {
      TGN_LOG_INFO(
          "TGUFBuilder: Using hardcoded edge splits (Val Start: {}, Test "
          "Start: {})",
          header_.val_start, header_.test_start);
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
    store.fst_length = mapped_bytes_;
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
    if (ftruncate(fd, mapped_bytes_) != 0) {
      close(fd);
      throw std::runtime_error("Failed to set file size (macOS)");
    }
#else
    if (posix_fallocate(fd, 0, mapped_bytes_) != 0) {
      close(fd);
      throw std::runtime_error("Failed to allocate disk space (Linux)");
    }
#endif

    base_ptr_ =
        mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base_ptr_ == MAP_FAILED) {
      throw std::runtime_error("Mmap for builder failed");
    }
  }

  ~Impl() {
    if (!finalized_) {
      munmap(base_ptr_, mapped_bytes_);
    }
  }

  auto to_mmap(std::uint64_t offset, std::size_t start_idx,
               std::size_t element_size, const torch::Tensor& t) const -> void {
    auto t_contiguous = t.contiguous();
    TORCH_CHECK(t_contiguous.nbytes() == t.size(0) * element_size,
                "Tensor size mismatch with expected element size");
    auto* dst = static_cast<std::uint8_t*>(base_ptr_) + offset +
                (start_idx * element_size);
    std::memcpy(dst, t_contiguous.data_ptr(), t_contiguous.nbytes());
  }
};

TGUFBuilder::TGUFBuilder(const TGUFSchema& schema)
    : impl_(std::make_unique<Impl>(schema)) {}
TGUFBuilder::~TGUFBuilder() = default;

auto TGUFBuilder::append_edges(const Batch& batch) const -> void {
  if (impl_->finalized_) {
    throw std::runtime_error(
        "TGUFBuilder::append_edges: Cannot append to a finalized file.");
  }

  auto count = batch.src.size(0);
  if (count == 0) {
    return;
  }
  TGN_LOG_DEBUG("TGUFBuilder: Appending {} edges to TGUF file", count);

  if (impl_->written_edges_ + count > impl_->schema_.edge_capacity) {
    throw std::runtime_error(
        "TGUFBuilder::append_edges: Overflow. Attempting to write " +
        std::to_string(count) + " edges, but only " +
        std::to_string(impl_->schema_.edge_capacity - impl_->written_edges_) +
        " slots remain.");
  }
  if (batch.msg.size(1) != static_cast<std::int64_t>(impl_->header_.msg_dim)) {
    throw std::invalid_argument(
        "TGUFBuilder::append_edges: Message dimension mismatch. Expected " +
        std::to_string(impl_->header_.msg_dim) + ", got " +
        std::to_string(batch.msg.size(1)));
  }
  impl_->to_mmap(impl_->header_.src_offset, impl_->written_edges_,
                 sizeof(std::int64_t), batch.src);
  impl_->to_mmap(impl_->header_.dst_offset, impl_->written_edges_,
                 sizeof(std::int64_t), batch.dst);
  impl_->to_mmap(impl_->header_.time_offset, impl_->written_edges_,
                 sizeof(std::int64_t), batch.time);
  impl_->to_mmap(impl_->header_.msg_offset, impl_->written_edges_,
                 impl_->header_.msg_dim * sizeof(float), batch.msg);

  if (impl_->header_.negatives_per_edge > 0) {
    if (!batch.neg_dst.has_value()) {
      throw std::invalid_argument(
          "TGUFBuilder::append_edges: neg_dst is required for this TGUF "
          "schema.");
    }
    if (batch.neg_dst->size(1) !=
        static_cast<std::int64_t>(impl_->header_.negatives_per_edge)) {
      throw std::invalid_argument(
          "TGUFBuilder::append_edges: Negative destination dimension mismatch. "
          "Expected " +
          std::to_string(impl_->header_.negatives_per_edge) + ", got " +
          std::to_string(batch.neg_dst->size(1)));
    }
  }

  if (impl_->header_.neg_dst_offset > 0 &&
      impl_->schema_.negatives_capacity > 0) {
    // The global index where negatives MUST start to fill the suffix
    const auto neg_global_start =
        impl_->schema_.edge_capacity - impl_->schema_.negatives_capacity;

    // Check if this batch overlaps with the negative storage zone
    if (impl_->written_edges_ + count > neg_global_start) {
      if (!batch.neg_dst.has_value()) {
        throw std::invalid_argument(
            "TGUFBuilder: Batch overlaps negative zone but no neg_dst "
            "provided.");
      }

      // Calculate which part of the batch falls into the negative zone
      const auto overlap_start =
          std::max(impl_->written_edges_, neg_global_start);
      const auto num_to_write = impl_->written_edges_ + count - overlap_start;

      // Determine where to read from the input tensor
      // If the input is 'compact' (just the negatives), we offset by how much
      // we've already written If the input is 'full' (aligned with batch.src),
      // we offset by (overlap_start - current_edge_idx)
      const auto is_compact = batch.neg_dst->size(0) < count;
      const auto read_offset = is_compact
                                   ? impl_->written_negatives_
                                   : (overlap_start - impl_->written_edges_);
      const auto write_idx = overlap_start - neg_global_start;

      impl_->to_mmap(
          impl_->header_.neg_dst_offset, write_idx,
          impl_->header_.negatives_per_edge * sizeof(std::int64_t),
          batch.neg_dst->slice(
              0, static_cast<std::int64_t>(read_offset),
              static_cast<std::int64_t>(read_offset + num_to_write)));

      impl_->written_negatives_ += num_to_write;
    }
  }
  impl_->written_edges_ += count;
}

auto TGUFBuilder::append_labels(const torch::Tensor& n_id,
                                const torch::Tensor& time,
                                const torch::Tensor& target) const -> void {
  if (impl_->finalized_) {
    throw std::runtime_error(
        "TGUFBuilder::append_labels: Cannot append labels to a finalized "
        "file.");
  }

  const auto count = n_id.size(0);
  if (count == 0) {
    return;
  }
  TGN_LOG_DEBUG("TGUFBuilder: Appending {} labels to TGUF file", count);

  if (impl_->written_labels_ + count > impl_->schema_.label_capacity) {
    throw std::runtime_error(
        "TGUFBuilder::append_labels: Overflow. Writing " +
        std::to_string(count) + " labels, but only " +
        std::to_string(impl_->schema_.label_capacity - impl_->written_labels_) +
        " slots remain.");
  }

  if (target.size(1) != static_cast<std::int64_t>(impl_->header_.label_dim)) {
    throw std::invalid_argument(
        "TGUFBuilder::append_labels: Label dimension mismatch. Expected " +
        std::to_string(impl_->header_.label_dim) + ", got " +
        std::to_string(target.size(1)));
  }

  impl_->to_mmap(impl_->header_.label_n_id_offset, impl_->written_labels_,
                 sizeof(std::int64_t), n_id);
  impl_->to_mmap(impl_->header_.label_time_offset, impl_->written_labels_,
                 sizeof(std::int64_t), time);
  impl_->to_mmap(impl_->header_.label_target_offset, impl_->written_labels_,
                 impl_->header_.label_dim * sizeof(float), target);
  impl_->written_labels_ += count;
}

auto TGUFBuilder::finalize() -> void {
  if (impl_->finalized_) {
    return;
  }

  if (impl_->written_edges_ < impl_->schema_.edge_capacity) {
    TGN_LOG_WARN(
        "TGUFBuilder: Finalizing with fewer edges than declared ({} < {}). "
        "File will have unused padding.",
        impl_->written_edges_, impl_->schema_.edge_capacity);
  }
  if (impl_->written_labels_ < impl_->schema_.label_capacity) {
    TGN_LOG_WARN(
        "TGUFBuilder: Finalizing with fewer labels than declared ({} < {}). "
        "File will have unused padding.",
        impl_->written_labels_, impl_->schema_.label_capacity);
  }
  if (impl_->written_negatives_ < impl_->schema_.negatives_capacity) {
    TGN_LOG_WARN(
        "TGUFBuilder: Finalizing with fewer negatives than declared ({} < {}). "
        "File will have unused padding.",
        impl_->written_negatives_, impl_->schema_.negatives_capacity);
  }

  // Update header_ with counts (user might have wrote less than declared)
  impl_->header_.num_edges = impl_->written_edges_;
  impl_->header_.num_labels = impl_->written_labels_;
  impl_->header_.num_negatives = impl_->written_negatives_;
  std::memcpy(impl_->base_ptr_, &impl_->header_, sizeof(TGUFHeader));

  msync(impl_->base_ptr_, impl_->mapped_bytes_, MS_SYNC);
  munmap(impl_->base_ptr_, impl_->mapped_bytes_);
  impl_->base_ptr_ = nullptr;
  impl_->finalized_ = true;

  TGN_LOG_INFO(
      "TGUFBuilder: Finalized to {} (Total edges: {}, Total labels: {}, Total "
      "negatives: {})",
      impl_->schema_.path, impl_->header_.num_edges, impl_->header_.num_labels,
      impl_->header_.num_negatives);
}
}  // namespace tgn
