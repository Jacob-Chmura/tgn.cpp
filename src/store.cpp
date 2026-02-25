#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "lib.h"

namespace tgn {

class InMemoryTGStore final : public TGStore {
 private:
  struct RandomNegSampler {
    std::int64_t min_id;
    std::int64_t max_id;

    [[nodiscard]] auto sample(std::int64_t n) const -> torch::Tensor {
      return torch::randint(min_id, max_id, {n, 1}, torch::kLong);
    }
  };

 public:
  explicit InMemoryTGStore(const InMemoryTGStoreOptions& opts)
      : src_(opts.src),
        dst_(opts.dst),
        t_(opts.t),
        msg_(opts.msg),
        neg_dst_(opts.neg_dst),
        num_edges_(static_cast<std::size_t>(src_.size(0))),
        num_nodes_(num_edges_ > 0
                       ? 1 + std::max(src_.max().item<std::int64_t>(),
                                      dst_.max().item<std::int64_t>())
                       : 0),
        msg_dim_(static_cast<std::size_t>(msg_.size(1))),
        train_(0,
               opts.val_start.value_or(opts.test_start.value_or(num_edges_))),
        val_(opts.val_start.value_or(opts.test_start.value_or(num_edges_)),
             opts.test_start.value_or(num_edges_)),
        test_(opts.test_start.value_or(num_edges_), num_edges_) {
    TORCH_CHECK(src_.dim() == 1, "src must be 1D");
    TORCH_CHECK(dst_.dim() == 1 && dst_.size(0) == src_.size(0),
                "dst must be 1D [n]");
    TORCH_CHECK(t_.dim() == 1 && t_.size(0) == src_.size(0),
                "t must be 1D [n]");
    TORCH_CHECK(msg_.dim() == 2 && msg_.size(0) == src_.size(0),
                "msg must be 2D [n, d]");
    TORCH_CHECK(!src_.is_floating_point(), "src must be integral");
    TORCH_CHECK(!dst_.is_floating_point(), "dst must be integral");

    if (train_.size() > 0) {
      const auto train_dst =
          dst_.slice(0, 0, static_cast<std::int64_t>(train_.end()));
      sampler_ = RandomNegSampler{
          .min_id = train_dst.min().item<std::int64_t>(),
          .max_id = train_dst.max().item<std::int64_t>(),
      };
    }

    if (neg_dst_.has_value()) {
      const auto& neg = neg_dst_.value();
      TORCH_CHECK(neg.dim() == 2, "neg_dst must be 2D [n, m]");
      TORCH_CHECK(neg.size(0) == static_cast<std::int64_t>(num_edges_),
                  "neg_dst row count must match num_edges");
      TORCH_CHECK(!neg.is_floating_point(), "neg_dst must be integral");
      TORCH_CHECK(neg.max().item<std::int64_t>() <
                      static_cast<std::int64_t>(num_nodes_),
                  "neg_dst contains IDs outside the range of src/dst");
    }

    if (opts.label_n_id.has_value()) {
      TORCH_CHECK(opts.label_t.has_value() && opts.label_y_true.has_value(),
                  "Missing label tensors");

      const auto l_t = opts.label_t.value();
      const auto l_n = opts.label_n_id.value();
      const auto l_y = opts.label_y_true.value();

      // Find unique timestamps in label_t (assumed sorted)
      const auto [unique_ts, inverse_indices, counts] =
          torch::unique_consecutive(l_t, /*return_inverse=*/true,
                                    /*return_counts=*/true);

      std::int64_t offset = 0;
      for (auto i = 0; i < unique_ts.size(0); ++i) {
        const auto count = counts[i].item<int64_t>();

        // Group the nodes/labels for this timestamp
        label_events_.push_back(
            LabelEvent{.n_id = l_n.slice(0, offset, offset + count),
                       .y_true = l_y.slice(0, offset, offset + count)});

        // Find the Edge Stop Index (first edge index where t >= label_time)
        const auto event_time = unique_ts[i].item<float>();
        auto it =
            std::lower_bound(t_.data_ptr<float>(),
                             t_.data_ptr<float>() + num_edges_, event_time);
        stop_e_ids_.push_back(std::distance(t_.data_ptr<float>(), it));

        offset += count;
      }

      // Determine Label Event Ranges (Interleave with Edge Splits)
      train_label_ = calculate_label_range(0.0, get_edge_time(train_.end()));
      val_label_ = calculate_label_range(get_edge_time(val_.start()),
                                         get_edge_time(val_.end()));
      test_label_ = calculate_label_range(get_edge_time(test_.start()),
                                          get_edge_time(test_.end()));
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

  auto calculate_label_range(std::int64_t start_t, std::int64_t end_t)
      -> Range {
    if (label_events_.empty() || start_t >= end_t) {
      return {0, 0};
    }

    // This assumes we stored timestamps somewhere or we can infer them
    // For now, use stop_e_ids_ to find labels that fall within edge window
    std::size_t start_idx = 0;
    std::size_t end_idx = 0;
    auto found_start = false;

    for (auto i = 0; i < stop_e_ids_.size(); ++i) {
      auto e_pos = stop_e_ids_[i];
      // If the stop_idx for this label event is within the edge split
      // TODO(kuba): refine this
      if (e_pos >= find_e_idx_at_time(start_t) &&
          e_pos <= find_e_idx_at_time(end_t)) {
        if (!found_start) {
          start_idx = i;
          found_start = true;
        }
        end_idx = i + 1;
      }
    }
    return {start_idx, end_idx};
  }

  [[nodiscard]] auto get_edge_time(std::int64_t idx) const -> std::int64_t {
    if (idx >= num_edges_) {
      return (num_edges_ > 0) ? t_[num_edges_ - 1].item<std::int64_t>() + 1 : 0;
    }
    return t_[idx].item<std::int64_t>();
  }

  [[nodiscard]] auto find_e_idx_at_time(std::int64_t time) const
      -> std::size_t {
    auto* it = std::lower_bound(t_.data_ptr<std::int64_t>(),
                                t_.data_ptr<std::int64_t>() + num_edges_, time);
    return std::distance(t_.data_ptr<std::int64_t>(), it);
  }
};

auto make_store(const InMemoryTGStoreOptions& opts)
    -> std::shared_ptr<TGStore> {
  return std::make_shared<InMemoryTGStore>(opts);
}

}  // namespace tgn
