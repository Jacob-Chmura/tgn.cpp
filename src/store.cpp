#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "lib.h"

namespace tgn {

class DummyTGStore final : public TGStore {
 public:
  explicit DummyTGStore() {
    n_edges_ = 100;
    msg_dim_ = 7;
  }

  [[nodiscard]] auto num_edges() const -> std::size_t override {
    return n_edges_;
  }
  [[nodiscard]] auto num_nodes() const -> std::size_t override { return 1000; };
  [[nodiscard]] auto msg_dim() const -> std::size_t override {
    return msg_dim_;
  };

  [[nodiscard]] auto get_batch(std::size_t start, std::size_t batch_size) const
      -> Batch override {
    const auto end = std::min(start + batch_size, n_edges_);
    const auto current_batch_size = static_cast<std::int64_t>(end - start);
    return Batch{
        .src = torch::randint(0, 5, {current_batch_size}),
        .dst = torch::randint(0, 5, {current_batch_size}),
        .t = torch::arange(static_cast<std::int64_t>(start),
                           static_cast<std::int64_t>(end), torch::kLong),
        .msg = torch::zeros({current_batch_size, msg_dim_}),
        .neg_dst = torch::randint(0, 5, {current_batch_size}),
    };
  }

  [[nodiscard]] auto get_t(const torch::Tensor& e_id) const
      -> torch::Tensor override {
    return torch::rand({e_id.size(0)});
  }

  [[nodiscard]] auto get_msg(const torch::Tensor& e_id) const
      -> torch::Tensor override {
    return torch::rand({e_id.size(0), msg_dim_});
  }

 private:
  std::size_t n_edges_{};
  std::int64_t msg_dim_{};
  torch::Tensor src_{}, dst_{}, t_{}, neg_dst_{};
};

auto make_store(const DummyTGStoreOptions& opts) -> std::unique_ptr<TGStore> {
  return std::make_unique<DummyTGStore>();
}

}  // namespace tgn
