#include <gtest/gtest.h>
#include <torch/torch.h>

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
