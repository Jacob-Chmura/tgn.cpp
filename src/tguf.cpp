#include "tguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <torch/types.h>
#include <unistd.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "tgn.h"

namespace tgn {

struct TGUFBuilder::Impl {
  std::string path{};
  TGUFHeader header;
  bool finalized = false;

  std::size_t declared_edges{};
  std::size_t declared_labels{};
  std::size_t written_edges{};
  std::size_t written_labels{};

  void* base_ptr = nullptr;
  std::size_t total_mapped_size{};

  Impl(std::string p, std::size_t n_edges, std::size_t n_labels,
       std::size_t m_dim, std::size_t l_dim, std::size_t n_neg,
       std::optional<std::size_t> val_start,
       std::optional<std::size_t> test_start)
      : path(std::move(p)), declared_edges(n_edges), declared_labels(n_labels) {
    header.msg_dim = m_dim;
    header.label_dim = l_dim;
    header.n_neg = n_neg;
    header.val_start = val_start.value_or(0);
    header.test_start = test_start.value_or(0);

    auto align = [](std::size_t size) {
      return (size + TGUF_PAGE - 1) & ~(TGUF_PAGE - 1);
    };

    header.src_offset = TGUF_PAGE;  // Header gets its own page
    header.dst_offset =
        header.src_offset + align(n_edges * sizeof(std::int64_t));
    header.t_offset = header.dst_offset + align(n_edges * sizeof(std::int64_t));
    header.msg_offset = header.t_offset + align(n_edges * sizeof(std::int64_t));

    std::size_t last_offset =
        header.msg_offset + align(n_edges * m_dim * sizeof(float));

    if (n_neg > 0) {
      header.neg_dst_offset = last_offset;
      last_offset += align(n_edges * n_neg * sizeof(std::int64_t));
    }

    if (n_labels > 0) {
      header.label_n_id_offset = last_offset;
      header.label_t_offset =
          header.label_n_id_offset + align(n_labels * sizeof(std::int64_t));
      header.label_y_true_offset =
          header.label_t_offset + align(n_labels * sizeof(std::int64_t));
      last_offset =
          header.label_y_true_offset + align(n_labels * l_dim * sizeof(float));
    }

    total_mapped_size = last_offset;

    auto fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) {
      throw std::runtime_error("Failed to create file");
    }
    if (posix_fallocate(fd, 0, total_mapped_size) != 0) {
      close(fd);
      throw std::runtime_error("Failed to allocate disk space");
    }

    base_ptr = mmap(nullptr, total_mapped_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
    close(fd);
    if (base_ptr == MAP_FAILED) {
      throw std::runtime_error("Mmap for builder failed");
    }
  }

  ~Impl() {
    if (!finalized) {
      munmap(base_ptr, total_mapped_size);
    }
  }

  auto to_mmap(std::uint64_t offset, std::size_t start_idx,
               std::size_t element_size, const torch::Tensor& t) const -> void {
    auto t_contiguous = t.contiguous();
    auto* dst = static_cast<std::uint8_t*>(base_ptr) + offset +
                (start_idx * element_size);
    std::memcpy(dst, t_contiguous.data_ptr(), t_contiguous.nbytes());
  }
};

TGUFBuilder::TGUFBuilder(const std::string& path, std::size_t n_edges,
                         std::size_t n_labels, std::size_t m_dim,
                         std::size_t l_dim, std::size_t n_neg,
                         std::optional<std::size_t> val_start,
                         std::optional<std::size_t> test_start)
    : impl_(std::make_unique<Impl>(path, n_edges, n_labels, m_dim, l_dim, n_neg,
                                   val_start, test_start)) {}
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
  if (impl_->written_edges + count > impl_->declared_edges) {
    throw std::runtime_error(
        "TGUFBuilder::append_edges: Overflow. Attempting to write " +
        std::to_string(count) + " edges, but only " +
        std::to_string(impl_->declared_edges - impl_->written_edges) +
        " slots remain.");
  }
  if (batch.msg.size(1) != static_cast<std::int64_t>(impl_->header.msg_dim)) {
    throw std::invalid_argument(
        "TGUFBuilder::append_edges: Message dimension mismatch. Expected " +
        std::to_string(impl_->header.msg_dim) + ", got " +
        std::to_string(batch.msg.size(1)));
  }
  if (impl_->header.n_neg > 0) {
    if (!batch.neg_dst.has_value()) {
      throw std::invalid_argument(
          "TGUFBuilder::append_edges: neg_dst is required for this TGUF "
          "schema.");
    }
    if (batch.neg_dst->size(1) !=
        static_cast<std::int64_t>(impl_->header.n_neg)) {
      throw std::invalid_argument(
          "TGUFBuilder::append_edges: Negative destination dimension mismatch. "
          "Expected " +
          std::to_string(impl_->header.n_neg) + ", got " +
          std::to_string(batch.neg_dst->size(1)));
    }
  }

  impl_->to_mmap(impl_->header.src_offset, impl_->written_edges,
                 sizeof(std::int64_t), batch.src);
  impl_->to_mmap(impl_->header.dst_offset, impl_->written_edges,
                 sizeof(std::int64_t), batch.dst);
  impl_->to_mmap(impl_->header.t_offset, impl_->written_edges,
                 sizeof(std::int64_t), batch.t);
  impl_->to_mmap(impl_->header.msg_offset, impl_->written_edges,
                 impl_->header.msg_dim * sizeof(float), batch.msg);

  if (batch.neg_dst.has_value() && impl_->header.neg_dst_offset > 0) {
    impl_->to_mmap(impl_->header.neg_dst_offset, impl_->written_edges,
                   impl_->header.n_neg * sizeof(std::int64_t), *batch.neg_dst);
  }
  impl_->written_edges += count;
}

auto TGUFBuilder::append_labels(const torch::Tensor& n_id,
                                const torch::Tensor& t,
                                const torch::Tensor& y_true) const -> void {
  if (impl_->finalized) {
    throw std::runtime_error(
        "TGUFBuilder::append_labels: Cannot append labels to a finalized "
        "file.");
  }

  const auto count = n_id.size(0);
  if (count == 0) {
    return;
  }
  if (impl_->written_labels + count > impl_->declared_labels) {
    throw std::runtime_error(
        "TGUFBuilder::append_labels: Overflow. Writing " +
        std::to_string(count) + " labels, but only " +
        std::to_string(impl_->declared_labels - impl_->written_labels) +
        " slots remain.");
  }

  if (y_true.size(1) != static_cast<std::int64_t>(impl_->header.label_dim)) {
    throw std::invalid_argument(
        "TGUFBuilder::append_labels: Label dimension mismatch. Expected " +
        std::to_string(impl_->header.label_dim) + ", got " +
        std::to_string(y_true.size(1)));
  }

  impl_->to_mmap(impl_->header.label_n_id_offset, impl_->written_labels,
                 sizeof(std::int64_t), n_id);
  impl_->to_mmap(impl_->header.label_t_offset, impl_->written_labels,
                 sizeof(std::int64_t), t);
  impl_->to_mmap(impl_->header.label_y_true_offset, impl_->written_labels,
                 impl_->header.label_dim * sizeof(float), y_true);
  impl_->written_labels += count;
}

auto TGUFBuilder::finalize() -> void {
  if (impl_->finalized) {
    return;
  }

  // Update header with counts (user might have wrote less than declared)
  impl_->header.num_edges = impl_->written_edges;
  impl_->header.num_labels = impl_->written_labels;
  std::memcpy(impl_->base_ptr, &impl_->header, sizeof(TGUFHeader));

  msync(impl_->base_ptr, impl_->total_mapped_size, MS_SYNC);
  munmap(impl_->base_ptr, impl_->total_mapped_size);
  impl_->base_ptr = nullptr;
  impl_->finalized = true;
}
}  // namespace tgn
