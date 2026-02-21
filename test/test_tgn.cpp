#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "lib.h"

TEST(TGNConfig, DefaultValues) {
  const tgn::TGNConfig cfg;
  EXPECT_EQ(cfg.embedding_dim, 100);
  EXPECT_EQ(cfg.memory_dim, 100);
  EXPECT_EQ(cfg.time_dim, 100);
  EXPECT_EQ(cfg.num_heads, 2);
  EXPECT_EQ(cfg.num_nbrs, 10);
  EXPECT_FLOAT_EQ(cfg.dropout, 0.1);
}

TEST(TGNConfig, CustomValues) {
  const tgn::TGNConfig cfg{.embedding_dim = 64, .memory_dim = 32};
  EXPECT_EQ(cfg.embedding_dim, 64);
  EXPECT_EQ(cfg.memory_dim, 32);
  EXPECT_EQ(cfg.time_dim, 100);
  EXPECT_EQ(cfg.num_heads, 2);
  EXPECT_EQ(cfg.num_nbrs, 10);
  EXPECT_FLOAT_EQ(cfg.dropout, 0.1);
}

class TGNTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::int64_t n = 10;
    const std::int64_t d = 8;
    auto opts = tgn::InMemoryTGStoreOptions{
        .src = torch::randint(0, 100, {n}, torch::kLong),
        .dst = torch::randint(0, 100, {n}, torch::kLong),
        .t = torch::linspace(0, 100, n).to(torch::kLong),
        .msg = torch::randn({n, d}),
        .neg_dst = std::nullopt};
    store = make_store(opts);

    cfg.embedding_dim = 32;
    cfg.memory_dim = 16;
    cfg.time_dim = 8;
  }

  tgn::TGNConfig cfg;
  std::shared_ptr<tgn::TGStore> store;
};

TEST_F(TGNTest, InitializationAndForwardShape) {
  tgn::TGN model(cfg, store);

  const auto src_ids = torch::tensor({1, 2, 3}, torch::kLong);
  const auto dst_ids = torch::tensor({4, 5, 6}, torch::kLong);

  const auto [z_src, z_dst] = model->forward(src_ids, dst_ids);

  EXPECT_EQ(z_src.size(1), cfg.embedding_dim);
  EXPECT_FALSE(torch::isnan(z_src).any().item<bool>());
  EXPECT_FALSE(torch::isnan(z_dst).any().item<bool>());
}

TEST_F(TGNTest, ResetStateClearsMemory) {
  tgn::TGN model(cfg, store);
  const auto ids = torch::tensor({10}, torch::kLong);

  const auto [z1] = model->forward(ids);
  EXPECT_FALSE(torch::isnan(z1).any().item<bool>());

  model->update_state(
      torch::tensor({10}, torch::kLong), torch::tensor({20}, torch::kLong),
      torch::tensor({1}, torch::kLong),
      torch::randn({1, static_cast<std::int64_t>(store->msg_dim())}));

  const auto [z2] = model->forward(ids);
  EXPECT_FALSE(torch::isnan(z2).any().item<bool>());
  EXPECT_FALSE(torch::allclose(z1, z2));

  model->reset_state();
  const auto [z3] = model->forward(ids);
  EXPECT_TRUE(torch::allclose(z1, z3));
  EXPECT_FALSE(torch::isnan(z3).any().item<bool>());
}

TEST_F(TGNTest, HandlesVariadicInputs) {
  tgn::TGN model(cfg, store);
  const auto n1 = torch::tensor({1}, torch::kLong);
  const auto n2 = torch::tensor({2, 3}, torch::kLong);
  const auto n3 = torch::tensor({4, 5, 6}, torch::kLong);

  // Ensure the variadic template handles different shapes
  const auto [z1, z2, z3] = model->forward(n1, n2, n3);

  EXPECT_EQ(z1.size(0), 1);
  EXPECT_EQ(z2.size(0), 2);
  EXPECT_EQ(z3.size(0), 3);
  EXPECT_FALSE(torch::isnan(z1).any().item<bool>());
  EXPECT_FALSE(torch::isnan(z2).any().item<bool>());
  EXPECT_FALSE(torch::isnan(z3).any().item<bool>());
}

TEST_F(TGNTest, GradientFlowAndNoNanGrads) {
  tgn::TGN model(cfg, store);
  const auto ids = torch::tensor({1, 2}, torch::kLong);

  const auto [z] = model->forward(ids);
  auto loss = z.sum();

  EXPECT_FALSE(torch::isnan(z).any().item<bool>());
  ASSERT_NO_THROW(loss.backward());
  ASSERT_GT(model->parameters().size(), 0);

  auto found_grad = false;
  for (const auto& p : model->parameters()) {
    if (p.requires_grad() && p.grad().defined()) {
      found_grad = true;
      EXPECT_FALSE(torch::isnan(p.grad()).any().item<bool>());
    }
  }
  EXPECT_TRUE(found_grad);
}

TEST_F(TGNTest, DetachMemoryStopsGradients) {
  tgn::TGN model(cfg, store);
  const auto ids = torch::tensor({1}, torch::kLong);

  const auto [z1] = model->forward(ids);
  EXPECT_FALSE(torch::isnan(z1).any().item<bool>());
  model->detach_memory();

  const auto [z2] = model->forward(ids);
  EXPECT_FALSE(torch::isnan(z2).any().item<bool>());
  EXPECT_TRUE(z2.grad_fn() != nullptr);
}
