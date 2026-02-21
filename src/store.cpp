#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "lib.h"

namespace tgn {

class InMemoryTGStore final : public TGStore {
 public:
  explicit InMemoryTGStore(const InMemoryTGStoreOptions& opts)
      : src_(opts.src),
        dst_(opts.dst),
        t_(opts.t),
        msg_(opts.msg),
        neg_dst_(opts.neg_dst) {
    TORCH_CHECK(src_.dim() == 1, "src must be 1D [n]");
    TORCH_CHECK(dst_.dim() == 1 && dst_.size(0) == src_.size(0),
                "dst must be 1D [n]");
    TORCH_CHECK(t_.dim() == 1 && t_.size(0) == src_.size(0),
                "t must be 1D [n]");
    TORCH_CHECK(msg_.dim() == 2 && msg_.size(0) == src_.size(0),
                "msg must be 2D [n, d]");
    TORCH_CHECK(!src_.is_floating_point(), "src must be integral");
    TORCH_CHECK(!dst_.is_floating_point(), "dst must be integral");

    num_edges_ = static_cast<std::size_t>(src_.size(0));
    msg_dim_ = msg_.size(1);
    num_nodes_ = num_edges_ > 0 ? 1 + std::max(src_.max().item<std::int64_t>(),
                                               dst_.max().item<std::int64_t>())
                                : 0;

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

  [[nodiscard]] auto get_batch(std::size_t start, std::size_t batch_size) const
      -> Batch override {
    const auto end = std::min(start + batch_size, num_edges_);
    const auto s = static_cast<std::int64_t>(start);
    const auto e = static_cast<std::int64_t>(end);

    return Batch{.src = src_.slice(0, s, e),
                 .dst = dst_.slice(0, s, e),
                 .t = t_.slice(0, s, e),
                 .msg = msg_.slice(0, s, e),
                 .neg_dst = neg_dst_.has_value() ? std::optional<torch::Tensor>(
                                                       neg_dst_->slice(0, s, e))
                                                 : std::nullopt};
  }

  [[nodiscard]] auto gather_timestamps(const torch::Tensor& e_id) const
      -> torch::Tensor override {
    return t_.index_select(0, e_id.flatten());
  }

  [[nodiscard]] auto gather_msgs(const torch::Tensor& e_id) const
      -> torch::Tensor override {
    return msg_.index_select(0, e_id.flatten());
  }

 private:
  std::size_t num_nodes_{0};
  std::size_t num_edges_{0};
  std::int64_t msg_dim_{0};
  torch::Tensor src_, dst_, t_, msg_;
  std::optional<torch::Tensor> neg_dst_;
};

auto make_store(const InMemoryTGStoreOptions& opts)
    -> std::shared_ptr<TGStore> {
  return std::make_shared<InMemoryTGStore>(opts);
}

}  // namespace tgn
