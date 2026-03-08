#include <fcntl.h>
#include <sys/mman.h>
#include <torch/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "tgn.h"
#include "tguf.h"

namespace tgn {
namespace detail {

struct TGData {
  torch::Tensor src;
  torch::Tensor dst;
  torch::Tensor time;
  torch::Tensor msg;
  std::optional<torch::Tensor> neg_dst = std::nullopt;

  std::optional<torch::Tensor> label_n_id = std::nullopt;
  std::optional<torch::Tensor> label_time = std::nullopt;
  std::optional<torch::Tensor> label_target = std::nullopt;

  std::size_t negatives_start_e_id{};
  std::optional<std::size_t> val_start = std::nullopt;
  std::optional<std::size_t> test_start = std::nullopt;

  auto validate() const -> void {
    const auto n = src.size(0);

    TORCH_CHECK(src.device().is_cpu(), "src must be on CPU");
    TORCH_CHECK(dst.device().is_cpu(), "dst must be on CPU");
    TORCH_CHECK(time.device().is_cpu(), "time must be on CPU");
    TORCH_CHECK(msg.device().is_cpu(), "msg must be on CPU");

    TORCH_CHECK(src.scalar_type() == torch::kLong, "src must be torch::kLong");
    TORCH_CHECK(dst.scalar_type() == torch::kLong, "dst must be torch::kLong");
    TORCH_CHECK(time.scalar_type() == torch::kLong,
                "time must be torch::kLong");
    TORCH_CHECK(msg.scalar_type() == torch::kFloat32,
                "msg must be torch::kFloat32");

    TORCH_CHECK(src.dim() == 1, "src must be 1D");
    TORCH_CHECK(dst.dim() == 1 && dst.size(0) == n, "dst must be [num_edges]");
    TORCH_CHECK(time.dim() == 1 && time.size(0) == n, "t must be [n]");
    TORCH_CHECK(msg.dim() == 2 && msg.size(0) == n,
                "msg must be [num_edges, d]");

    if (neg_dst.has_value()) {
      TORCH_CHECK(neg_dst->device().is_cpu(), "neg_dst must be on CPU");
      TORCH_CHECK(neg_dst->scalar_type() == torch::kLong,
                  "neg_dst must be torch::Long");

      const auto num_nodes = n > 0
                                 ? 1 + std::max(src.max().item<std::int64_t>(),
                                                dst.max().item<std::int64_t>())
                                 : 0;
      TORCH_CHECK(
          neg_dst->dim() == 2 && neg_dst->size(0) == n - negatives_start_e_id,
          "neg_dst must be [num_edges, m]");
      TORCH_CHECK(neg_dst->max().item<std::int64_t>() < num_nodes,
                  "neg_dst contains IDs outside the range of src/dst");
    }

    if (label_n_id.has_value()) {
      TORCH_CHECK(
          label_time.has_value() && label_target.has_value(),
          "If label_n_id is provided, label_time and label_target must exist");

      TORCH_CHECK(label_n_id->device().is_cpu(), "label_n_id must be on CPU");
      TORCH_CHECK(label_time->device().is_cpu(), "label_time must be on CPU");
      TORCH_CHECK(label_target->device().is_cpu(),
                  "label_target must be on CPU");

      TORCH_CHECK(label_n_id->scalar_type() == torch::kLong,
                  "label_n_id must be torch::kLong");
      TORCH_CHECK(label_time->scalar_type() == torch::kLong,
                  "label_time must be torch::kLong");

      const auto n_labels = label_n_id->size(0);
      TORCH_CHECK(label_time->size(0) == n_labels, "label_time size mismatch");
      TORCH_CHECK(label_target->size(0) == n_labels,
                  "label_target size mismatch");
    }
  }

  [[nodiscard]] auto get_size_bytes() const -> std::size_t {
    auto bytes = src.nbytes() + dst.nbytes() + time.nbytes() + msg.nbytes();
    if (neg_dst.has_value()) {
      bytes += neg_dst->nbytes();
    }
    if (label_n_id.has_value()) {
      bytes += label_n_id->nbytes();
      bytes += label_time->nbytes();
      bytes += label_target->nbytes();
    }
    return bytes;
  }
};

class TGStoreImpl final : public TGStore {
 private:
  struct RandomNegSampler {
    std::int64_t min_id;
    std::int64_t max_id;

    [[nodiscard]] auto sample(std::int64_t n) const -> torch::Tensor {
      return torch::randint(min_id, max_id, {n, 1}, torch::kLong);
    }
  };

 public:
  explicit TGStoreImpl(TGData data)
      : src_(std::move(data.src)),
        dst_(std::move(data.dst)),
        t_(std::move(data.time)),
        msg_(std::move(data.msg)),
        neg_dst_(std::move(data.neg_dst)),
        num_edges_(static_cast<std::size_t>(src_.size(0))),
        num_nodes_(num_edges_ > 0
                       ? 1 + std::max(src_.max().item<std::int64_t>(),
                                      dst_.max().item<std::int64_t>())
                       : 0),
        msg_dim_(static_cast<std::size_t>(msg_.size(1))),
        label_dim_(static_cast<std::size_t>(
            data.label_target.has_value() ? data.label_target->size(1) : 0)),
        negatives_start_e_id_(data.negatives_start_e_id),
        train_(0,
               data.val_start.value_or(data.test_start.value_or(num_edges_))),
        val_(data.val_start.value_or(data.test_start.value_or(num_edges_)),
             data.test_start.value_or(num_edges_)),
        test_(data.test_start.value_or(num_edges_), num_edges_) {
    TGN_LOG_INFO("TGStore: Loaded {} edges ({} nodes, msg_dim: {})", num_edges_,
                 num_nodes_, msg_dim_);
    if (neg_dst_.has_value()) {
      TGN_LOG_INFO("TGStore: Pre-computed negatives found ({} negatives/edge)",
                   neg_dst_->size(1));
    }
    TGN_LOG_INFO("TGStore: Edge Splits Train[{}:{}] Val[{}:{}] Test[{}:{}]",
                 train_.start(), train_.end(), val_.start(), val_.end(),
                 test_.start(), test_.end());

    if (train_.size() > 0) {
      const auto train_dst =
          dst_.slice(0, 0, static_cast<std::int64_t>(train_.end()));
      sampler_ = RandomNegSampler{
          .min_id = train_dst.min().item<std::int64_t>(),
          .max_id = train_dst.max().item<std::int64_t>(),
      };
      TGN_LOG_INFO("TGStore: Using RandomNegSampler w/ sample range: [{}, {}]",
                   sampler_->min_id, sampler_->max_id);
    }

    if (data.label_n_id.has_value()) {
      const auto label_time = std::move(data.label_time.value());
      const auto label_n_id = std::move(data.label_n_id.value());
      const auto label_target = std::move(data.label_target.value());

      // Find unique timestamps in label_time (assumed sorted)
      const auto [unique_ts, inverse_indices, counts] =
          torch::unique_consecutive(label_time, /*return_inverse=*/true,
                                    /*return_counts=*/true);

      auto find_e_id_at_time = [&](std::int64_t time) -> std::size_t {
        auto* it =
            std::lower_bound(t_.data_ptr<std::int64_t>(),
                             t_.data_ptr<std::int64_t>() + num_edges_, time);
        return std::distance(t_.data_ptr<std::int64_t>(), it);
      };

      std::int64_t offset = 0;
      for (auto i = 0; i < unique_ts.size(0); ++i) {
        const auto count = counts[i].item<std::int64_t>();
        const auto event_time = unique_ts[i].item<std::int64_t>();

        // Group the nodes/labels for this timestamp
        label_events_.push_back(LabelEvent{
            .n_id = label_n_id.slice(0, offset, offset + count),
            .target = label_target.slice(0, offset, offset + count)});

        // Find the Edge Stop Index (first edge index where t >= label_time)
        stop_e_ids_.push_back(find_e_id_at_time(event_time));

        offset += count;
      }

      auto get_edge_time = [&](std::size_t e_id) -> std::int64_t {
        if (e_id >= num_edges_) {
          return (num_edges_ > 0) ? t_[-1].item<std::int64_t>() + 1 : 0;
        }
        return t_[static_cast<std::int64_t>(e_id)].item<std::int64_t>();
      };

      auto calculate_label_range = [&](std::int64_t start_t,
                                       std::int64_t end_t) -> IndexRange {
        if (label_events_.empty() || start_t >= end_t) {
          return IndexRange{};
        }

        // The 'round-trip' from train split edge start/end indices to
        // timestamps and then back to start/end e_pos is required since TGB
        // split boundaries may occur on a 'single time unit'. In order to
        // reproduce, node events must be processed as the same split.
        //
        // Ref:
        // https://github.com/tgm-team/tgm/blob/72c8bf9/tgm/data/dg_data.py#L1034-L1036
        // E.g. Handles TGB node label offset: tgbn-trade validation starts at
        // 2010 while first node event batch starts at 2009.
        const auto start_e_pos = find_e_id_at_time(start_t);
        const auto end_e_pos = find_e_id_at_time(end_t);

        // First label event that occurs at or after the start_e_pos
        auto it_start = std::ranges::lower_bound(stop_e_ids_, start_e_pos);

        // First label event that occurs after the end_e_pos
        auto it_end = std::ranges::upper_bound(stop_e_ids_, end_e_pos);

        auto start = static_cast<std::size_t>(
            std::distance(stop_e_ids_.begin(), it_start));
        auto end = static_cast<std::size_t>(
            std::distance(stop_e_ids_.begin(), it_end));

        TGN_LOG_INFO(
            "TGStore: Mapped Label Time [{}, {}] -> Edges [{}, {}] -> Label "
            "Split "
            "[{}:{}]",
            start_t, end_t, start_e_pos, end_e_pos, start, end);

        return IndexRange{start, end};
      };

      TGN_LOG_INFO(
          "TGStore: Node labels active (label_dim: {}, total_events: {})",
          label_dim_, label_events_.size());

      const auto train_start_time = 0;
      const auto train_end_time = get_edge_time(train_.end());
      const auto val_start = get_edge_time(val_.start());
      const auto val_end_time = get_edge_time(val_.end());
      const auto test_start = get_edge_time(test_.start());
      const auto test_end_time = get_edge_time(test_.end());

      // Determine Label Event Ranges (Interleave with Edge Splits)
      train_label_ = calculate_label_range(train_start_time, train_end_time);
      val_label_ = calculate_label_range(val_start, val_end_time);
      test_label_ = calculate_label_range(test_start, test_end_time);

      TGN_LOG_INFO("TGStore: Label Splits Train[{}:{}] Val[{}:{}] Test[{}:{}]",
                   train_label_.start(), train_label_.end(), val_label_.start(),
                   val_label_.end(), test_label_.start(), test_label_.end());
    }
  }

  [[nodiscard]] auto edge_count() const -> std::size_t override {
    return num_edges_;
  }
  [[nodiscard]] auto node_count() const -> std::size_t override {
    return num_nodes_;
  }
  [[nodiscard]] auto msg_dim() const -> std::size_t override {
    return msg_dim_;
  }
  [[nodiscard]] auto label_dim() const -> std::size_t override {
    return label_dim_;
  }
  [[nodiscard]] auto train_split() const -> IndexRange override {
    return train_;
  }
  [[nodiscard]] auto val_split() const -> IndexRange override { return val_; }
  [[nodiscard]] auto test_split() const -> IndexRange override { return test_; }
  [[nodiscard]] auto train_label_split() const -> IndexRange override {
    return train_label_;
  }
  [[nodiscard]] auto val_label_split() const -> IndexRange override {
    return val_label_;
  }
  [[nodiscard]] auto test_label_split() const -> IndexRange override {
    return test_label_;
  }

  [[nodiscard]] auto get_batch(std::size_t start, std::size_t batch_size,
                               NegStrategy strategy = NegStrategy::None) const
      -> Batch override {
    const auto end = std::min(start + batch_size, num_edges_);
    const auto s = static_cast<std::int64_t>(start);
    const auto e = static_cast<std::int64_t>(end);

    std::optional<torch::Tensor> batch_neg = std::nullopt;

    if (strategy == NegStrategy::Random) {  // TODO(kuba): fix rng
      TGN_LOG_DEBUG("TGStore: get_batch [{}:{}] (NegStrategy::Random)", start,
                    end);
      TORCH_CHECK(sampler_.has_value(),
                  "Random sampling requested but sampler not initialized "
                  "(train split is empty)");
      batch_neg = sampler_->sample(e - s);
    } else if (strategy == NegStrategy::PreComputed) {
      TGN_LOG_DEBUG("TGStore: get_batch [{}:{}] (NegStrategy::PreComputed)",
                    start, end);
      TORCH_CHECK(neg_dst_.has_value(),
                  "NegStrategy::PreComputed requested but no neg_dst tensor "
                  "available");
      if (s < negatives_start_e_id_) {
        throw std::runtime_error(
            "Attempted to access pre-computed negatives at index " +
            std::to_string(s) + " but negative storage starts at " +
            std::to_string(negatives_start_e_id_));
      }
      batch_neg = neg_dst_->slice(0, s - negatives_start_e_id_,
                                  e - negatives_start_e_id_);
    } else {
      TGN_LOG_DEBUG("TGStore: get_batch [{}:{}] (NegStrategy::None)", start,
                    end);
    }

    return Batch{.src = src_.slice(0, s, e),
                 .dst = dst_.slice(0, s, e),
                 .time = t_.slice(0, s, e),
                 .msg = msg_.slice(0, s, e),
                 .neg_dst = batch_neg};
  }

  [[nodiscard]] auto gather_timestamps(const torch::Tensor& e_id) const
      -> torch::Tensor override {
    return t_.index_select(0, e_id.flatten());
  }

  [[nodiscard]] auto gather_msgs(const torch::Tensor& e_id) const
      -> torch::Tensor override {
    return msg_.index_select(0, e_id.flatten());
  }

  [[nodiscard]] auto get_edge_cutoff_for_label_event(std::size_t l_id) const
      -> std::size_t override {
    return stop_e_ids_.at(l_id);
  }

  [[nodiscard]] auto get_label_event(std::size_t l_id) const
      -> LabelEvent override {
    return label_events_.at(l_id);
  }

 private:
  torch::Tensor src_, dst_, t_, msg_;
  std::optional<torch::Tensor> neg_dst_;

  std::size_t num_edges_{0};
  std::size_t num_nodes_{0};
  std::size_t msg_dim_{0};
  std::size_t label_dim_{0};
  std::size_t negatives_start_e_id_{0};

  IndexRange train_, val_, test_;
  IndexRange train_label_, val_label_, test_label_;
  std::optional<RandomNegSampler> sampler_;

  std ::vector<LabelEvent> label_events_;
  std ::vector<std::size_t> stop_e_ids_;
};

}  // namespace detail

[[nodiscard]] auto TGStore::from_memory(
    const Batch& edges, const std::optional<torch::Tensor>& label_n_id,
    const std::optional<torch::Tensor>& label_time,
    const std::optional<torch::Tensor>& label_target,
    std::optional<std::size_t> val_start, std::optional<std::size_t> test_start)
    -> std::shared_ptr<TGStore> {
  auto data = detail::TGData{.src = edges.src,
                             .dst = edges.dst,
                             .time = edges.time,
                             .msg = edges.msg,
                             .neg_dst = edges.neg_dst,
                             .label_n_id = label_n_id,
                             .label_time = label_time,
                             .label_target = label_target,
                             .val_start = val_start,
                             .test_start = test_start};
  data.validate();

  TGN_LOG_INFO("TGStore: Initialized from memory (~{:.2f} GiB allocated)",
               data.get_size_bytes() / (1024.0 * 1024.0 * 1024.0));
  return std::make_shared<detail::TGStoreImpl>(std::move(data));
}

[[nodiscard]] auto TGStore::from_tguf(const std::string& path,
                                      std::optional<std::size_t> val_start,
                                      std::optional<std::size_t> test_start)
    -> std::shared_ptr<TGStore> {
  TGN_LOG_INFO("TGStore: Mapping TGUF file: {}", path);
  auto fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Could not open file: " + path);
  }

  const auto file_size = std::filesystem::file_size(path);
  if (file_size < sizeof(TGUFHeader)) {
    close(fd);
    throw std::runtime_error("File too small to contain TGUF header");
  }

  auto* addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Mmap failed for: " + path);
  }

  auto* header = static_cast<TGUFHeader*>(addr);
  if (header->magic != TGUF_MAGIC) {
    munmap(addr, file_size);
    close(fd);
    throw std::runtime_error(
        "Invalid TGUF magic number. File is corrupted or wrong format.");
  }
  TGN_LOG_INFO("TGStore: TGUF v{:X} Header Parsed (Magic Ox{:X})",
               header->version, header->magic);

  // TGN training is mostly sequential per epoch.
  madvise(addr, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);

#ifdef MADV_HUGEPAGE
  // Hint to the kernel to use 2MB pages for this mapping.
  // This might reduces TLB misses during the gather phase.
  if (madvise(addr, file_size, MADV_HUGEPAGE) != 0) {
    TGN_LOG_WARN("TGStore: MADV_HUGEPAGE failed: {}", std::strerror(errno));
  }
  TGN_LOG_INFO("TGStore: MADV_HUGEPAGES is active");
#endif

  auto mapping_guard = std::shared_ptr<void>(addr, [file_size, fd](void* p) {
    munmap(p, file_size);
    close(fd);
  });

  auto* base = static_cast<char*>(addr);

  // Helper to create tensor views into the mmap
  auto mmap_tensor = [&](std::uint64_t offset, c10::IntArrayRef shape,
                         torch::Dtype dtype) {
    if (offset == 0) {
      return torch::Tensor();  // Return empty for optional field
    }
    if (shape.size() > 0 && shape[0] == 0) {
      return torch::empty(shape, torch::TensorOptions().dtype(dtype));
    }

    // Safety: ensure offset is within file bounds
    if (offset >= file_size) {
      throw std::runtime_error("TGUF offset out of bounds");
    }

    return torch::from_blob(
        base + offset, shape, [mapping_guard](void*) { /* Keep mmap alive */ },
        torch::TensorOptions().dtype(dtype));
  };

  auto resolve_split =
      [](const std::optional<std::size_t>& opt_val, std::uint64_t header_val,
         const std::string& name) -> std::optional<std::size_t> {
    if (opt_val.has_value()) {
      if (header_val != 0 && *opt_val != header_val) {
        TGN_LOG_WARN("TGUF: Overriding header {} ({}) with user value ({})",
                     name, header_val, *opt_val);
      }
      return opt_val;
    }
    return (header_val != 0)
               ? std::make_optional(static_cast<std::size_t>(header_val))
               : std::nullopt;
  };

  detail::TGData data{};
  data.val_start = resolve_split(val_start, header->val_start, "val_start");
  data.test_start = resolve_split(test_start, header->test_start, "test_start");

  const auto n_edges = static_cast<std::int64_t>(header->num_edges);
  const auto m_dim = static_cast<std::int64_t>(header->msg_dim);
  const auto negatives_per_edge =
      static_cast<std::int64_t>(header->negatives_per_edge);
  const auto n_labels = static_cast<std::int64_t>(header->num_labels);
  const auto l_dim = static_cast<std::int64_t>(header->label_dim);

  data.src = mmap_tensor(header->src_offset, {n_edges}, torch::kLong);
  data.dst = mmap_tensor(header->dst_offset, {n_edges}, torch::kLong);
  data.time = mmap_tensor(header->time_offset, {n_edges}, torch::kLong);
  data.msg = mmap_tensor(header->msg_offset, {n_edges, m_dim}, torch::kFloat32);

  if (header->neg_dst_offset > 0 && header->negatives_per_edge > 0) {
    data.negatives_start_e_id = header->negatives_start_e_id;
    const auto n_neg =
        n_edges - static_cast<std::int64_t>(header->negatives_start_e_id);
    if (n_neg > 0) {
      data.neg_dst = mmap_tensor(header->neg_dst_offset,
                                 {n_neg, negatives_per_edge}, torch::kLong);
    }
  }
  if (header->num_labels > 0) {
    data.label_n_id =
        mmap_tensor(header->label_n_id_offset, {n_labels}, torch::kLong);
    data.label_time =
        mmap_tensor(header->label_time_offset, {n_labels}, torch::kLong);
    data.label_target = mmap_tensor(header->label_target_offset,
                                    {n_labels, l_dim}, torch::kFloat32);
  }

  data.validate();
  TGN_LOG_INFO("TGStore: Initialized from TGUF (~{:.2f} GiB Memory-Mapped)",
               data.get_size_bytes() / (1024.0 * 1024.0 * 1024.0));
  return std::make_shared<detail::TGStoreImpl>(std::move(data));
}

}  // namespace tgn
