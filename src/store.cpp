#include <fcntl.h>
#include <sys/mman.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "lib.h"

namespace tgn {

class StaticTGStoreImpl final : public TGStore {
 private:
  struct RandomNegSampler {
    std::int64_t min_id;
    std::int64_t max_id;

    [[nodiscard]] auto sample(std::int64_t n) const -> torch::Tensor {
      return torch::randint(min_id, max_id, {n, 1}, torch::kLong);
    }
  };

 public:
  explicit StaticTGStoreImpl(TGData data)
      : src_(std::move(data.src)),
        dst_(std::move(data.dst)),
        t_(std::move(data.t)),
        msg_(std::move(data.msg)),
        neg_dst_(std::move(data.neg_dst)),
        num_edges_(static_cast<std::size_t>(src_.size(0))),
        num_nodes_(num_edges_ > 0
                       ? 1 + std::max(src_.max().item<std::int64_t>(),
                                      dst_.max().item<std::int64_t>())
                       : 0),
        msg_dim_(static_cast<std::size_t>(msg_.size(1))),
        train_(0,
               data.val_start.value_or(data.test_start.value_or(num_edges_))),
        val_(data.val_start.value_or(data.test_start.value_or(num_edges_)),
             data.test_start.value_or(num_edges_)),
        test_(data.test_start.value_or(num_edges_), num_edges_) {
    if (train_.size() > 0) {
      const auto train_dst =
          dst_.slice(0, 0, static_cast<std::int64_t>(train_.end()));
      sampler_ = RandomNegSampler{
          .min_id = train_dst.min().item<std::int64_t>(),
          .max_id = train_dst.max().item<std::int64_t>(),
      };
    }

    if (data.label_n_id.has_value()) {
      const auto label_t = std::move(data.label_t.value());
      const auto label_n_id = std::move(data.label_n_id.value());
      const auto label_y_true = std::move(data.label_y_true.value());

      // Find unique timestamps in label_t (assumed sorted)
      const auto [unique_ts, inverse_indices, counts] =
          torch::unique_consecutive(label_t, /*return_inverse=*/true,
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
            .y_true = label_y_true.slice(0, offset, offset + count)});

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
                                       std::int64_t end_t) -> Range {
        if (label_events_.empty() || start_t >= end_t) {
          return Range{};
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

        return Range{start, end};
      };

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
    }
  }

  [[nodiscard]] auto num_edges() const -> std::size_t override {
    return num_edges_;
  }
  [[nodiscard]] auto num_nodes() const -> std::size_t override {
    return num_nodes_;
  }
  [[nodiscard]] auto msg_dim() const -> std::size_t override {
    return msg_dim_;
  }
  [[nodiscard]] auto train_split() const -> Range override { return train_; }
  [[nodiscard]] auto val_split() const -> Range override { return val_; }
  [[nodiscard]] auto test_split() const -> Range override { return test_; }

  [[nodiscard]] auto train_label_split() const -> Range override {
    return train_label_;
  }
  [[nodiscard]] auto val_label_split() const -> Range override {
    return val_label_;
  }
  [[nodiscard]] auto test_label_split() const -> Range override {
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
      TORCH_CHECK(sampler_.has_value(),
                  "Random sampling requested but sampler not initialized "
                  "(train split is empty)");
      batch_neg = sampler_->sample(e - s);
    } else if (strategy == NegStrategy::PreComputed) {
      TORCH_CHECK(
          neg_dst_.has_value(),
          "NegStrategy::PreComputed requested but no neg_dst tensor available");
      batch_neg = neg_dst_->slice(0, s, e);
    }

    return Batch{.src = src_.slice(0, s, e),
                 .dst = dst_.slice(0, s, e),
                 .t = t_.slice(0, s, e),
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

  [[nodiscard]] auto get_stop_e_id_for_label_event(std::size_t l_id) const
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

  Range train_, val_, test_;
  Range train_label_, val_label_, test_label_;
  std::optional<RandomNegSampler> sampler_;

  std ::vector<LabelEvent> label_events_;
  std ::vector<std::size_t> stop_e_ids_;
};

[[nodiscard]] auto TGStore::from_memory(TGData data)
    -> std::shared_ptr<TGStore> {
  data.validate();
  return std::make_shared<StaticTGStoreImpl>(std::move(data));
}

[[nodiscard]] static auto from_disk(const TGDiskOptions& opts)
    -> std::shared_ptr<TGStore> {
  auto fd = open(opts.path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Could not open file for reading: " + opts.path);
  }

  const auto file_size = std::filesystem::file_size(opts.path);
  auto* addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("mmap read failed for: " + opts.path);
  }

  // Hint for the kernel for manages OS pages nicely
  madvice(addr, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);

  // Custom deleter to unmap store when tensors go out of scope
  auto mapping_guard = std::shared_ptr<void>(addr, [file_size, fd](void *p) {
    munmap(p, file_size);
    close(fd);
  };

  auto* header = static_cast<TGDiskHeader*>(addr);
  auto* base = static_cast<char*>(addr);

  auto mmap_tensor = [&](std::uint64_t offset, c10::IntArrayRef shape, torch::Dtype dtype){
    return torch::from_blob(
        base + offset, shape,
        [mapping_guard](void*) { /* mapping_guard capture keeps it alive */ },
        torch::TensorOptions().dtype(type));
  };

  TGData data{.val_start = opts.val_start, .test_start = opts.test_start};

  data.src = mmap_tensor(header->src_offset, {header->num_edges}, torch::kLong);
  data.dst = mmap_tensor(header->dst_offset, {header->num_edges}, torch::kLong);
  data.t = mmap_tensor(header->t_offset, {header->num_edges}, torch::kLong);
  data.msg = mmap_tensor(header->msg_offset, {header->num_edges, header->msg_dim}, torch::kFloat32);

  if (header->neg_dst_offset > 0) {
    data.neg_dst =
        mmap_tensor(header->neg_dst_offset, {header->num_edges, header->n_neg},
                    torch::kLong);
  }

  if (header->num_labels > 0) {
    data.label_n_id = mmap_tensor(header->label_n_id_offset,
                                  {header->num_labels}, torch::kLong);
    data.label_t =
        mmap_tensor(header->label_t_offset, {header->num_labels}, torch::kLong);
    data.label_y_true = mmap_tensor(header->label_y_true_offset, {header->num_labels, header->label_dim)}, torch::kFloat32);
  }

  data.validate();
  return std::make_shared<StaticTGStoreImpl>(std::move(data));
}

auto TGData::validate() const -> void {
  const auto n = src.size(0);

  TORCH_CHECK(src.dim() == 1, "src must be 1D");
  TORCH_CHECK(dst.dim() == 1 && dst.size(0) == n, "dst must be [num_edges]");
  TORCH_CHECK(t.dim() == 1 && t.size(0) == n, "t must be [n]");
  TORCH_CHECK(msg.dim() == 2 && msg.size(0) == n, "msg must be [num_edges, d]");

  TORCH_CHECK(!src.is_floating_point(), "src must be integral");
  TORCH_CHECK(!dst.is_floating_point(), "dst must be integral");
  TORCH_CHECK(!t.is_floating_point(), "timestamps must be integral");

  if (neg_dst.has_value()) {
    const auto num_nodes = n > 0 ? 1 + std::max(src.max().item<std::int64_t>(),
                                                dst.max().item<std::int64_t>())
                                 : 0;
    const auto& neg = neg_dst.value();
    TORCH_CHECK(neg.dim() == 2 && neg.size(0) == n,
                "neg_dst must be [num_edges, m]");
    TORCH_CHECK(!neg.is_floating_point(), "neg_dst must be integral");
    TORCH_CHECK(neg.max().item<std::int64_t>() < num_nodes,
                "neg_dst contains IDs outside the range of src/dst");
  }

  if (label_n_id.has_value()) {
    TORCH_CHECK(
        label_t.has_value() && label_y_true.has_value(),
        "If label_n_id is provided, label_t and label_y_true must exist");

    const auto n_labels = label_n_id->size(0);
    TORCH_CHECK(label_t->size(0) == n_labels, "label_t size mismatch");
    TORCH_CHECK(label_y_true->size(0) == n_labels,
                "label_y_true size mismatch");
  }
}

}  // namespace tgn
