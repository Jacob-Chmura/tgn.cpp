#include <gtest/gtest.h>
#include <torch/types.h>

#include "scatter_ops.h"

TEST(ScatterOps, ScatterMax) {
  const auto src = torch::tensor({1.0, 4.0, 2.0, 10.0, 5.0});
  const auto index = torch::tensor({0, 0, 1, 1, 1}, torch::kLong);

  const auto out = tgn::scatter_max(src, index, 2);
  const auto expected = torch::tensor({4.0, 10.0});

  EXPECT_TRUE(out.allclose(expected));
}

TEST(ScatterOps, ScatterMaxWithTie) {
  const auto src = torch::tensor({7.0, 7.0, 7.0});
  const auto index = torch::tensor({0, 0, 0}, torch::kLong);

  const auto out = tgn::scatter_max(src, index, 1);
  EXPECT_TRUE(out.allclose(torch::tensor({7.0})));
}

TEST(ScatterOps, ScatterSoftmax) {
  // Group 0: [0, 0] -> [0.5, 0.5]
  // Group 1: [0, log(3)] -> [0.25, 0.75]
  const auto src = torch::tensor({0.0, 0.0, 0.0, std::log(3.0)});
  const auto index = torch::tensor({0, 0, 1, 1}, torch::kLong);

  const auto out = tgn::scatter_softmax(src, index, 2);
  const auto expected = torch::tensor({0.5, 0.5, 0.25, 0.75});

  EXPECT_TRUE(out.allclose(expected));
}

TEST(ScatterOps, ScatterSoftmaxNumericalStability) {
  // Inputs that would overflow exp() without the internal max subtraction
  const auto src = torch::tensor({1000.0, 1000.0, -1000.0, -1000.0});
  const auto index = torch::tensor({0, 0, 1, 1}, torch::kLong);

  const auto out = tgn::scatter_softmax(src, index, 2);

  EXPECT_FALSE(out.isnan().any().item<bool>());
  EXPECT_TRUE(out.allclose(torch::tensor({0.5, 0.5, 0.5, 0.5})));
}

TEST(ScatterOps, ScatterArgmax) {
  const auto src = torch::tensor({1.0, 10.0, 5.0, 2.0});
  const auto index = torch::tensor({0, 0, 1, 1}, torch::kLong);

  const auto out = tgn::scatter_argmax(src, index, 2);
  const auto expected = torch::tensor({1, 2}, torch::kLong);

  EXPECT_TRUE(out.equal(expected));
}

TEST(ScatterOps, ScatterArgmaxWithTie) {
  // In TGN, argmax is used on timestamps. This confirms last index wins
  // on a tie.
  const auto src = torch::tensor({10.0, 10.0, 5.0, 5.0});
  const auto index = torch::tensor({0, 0, 1, 1}, torch::kLong);

  const auto out = tgn::scatter_argmax(src, index, 2);

  // Node 0: max 10.0 at indices 0, 1 -> returns 1
  // Node 1: max 5.0 at indices 2, 3 -> returns 3
  const auto expected = torch::tensor({1, 3}, torch::kLong);
  EXPECT_TRUE(out.equal(expected));
}

TEST(ScatterOps, ScatterMax2D) {
  const auto src = torch::tensor({
      {1.0, 10.0},  // Edge 0 (Node 0)
      {5.0, 2.0},   // Edge 1 (Node 0)
      {3.0, 3.0}    // Edge 2 (Node 1)
  });
  const auto index = torch::tensor({0, 0, 1}, torch::kLong);

  const auto out = tgn::scatter_max(src, index, 2);

  // Node 0: max({1, 5}, {10, 2}) -> {5.0, 10.0}
  // Node 1: max({3}, {3})         -> {3.0, 3.0}
  const auto expected = torch::tensor({{5.0, 10.0}, {3.0, 3.0}});

  EXPECT_EQ(out.dim(), 2);
  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_TRUE(out.allclose(expected));
}

TEST(ScatterOps, ScatterAdd2D) {
  const auto src = torch::tensor({
      {1.0, 1.0, 1.0},  // Edge 0 (Node 0)
      {2.0, 2.0, 2.0},  // Edge 1 (Node 1)
      {3.0, 3.0, 3.0}   // Edge 2 (Node 0)
  });
  const auto index = torch::tensor({0, 1, 0}, torch::kLong);

  const auto out = tgn::scatter_add(src, index, 2);

  // Node 0: Sum(Edge 0, Edge 2) -> {4.0, 4.0, 4.0}
  // Node 1: Sum(Edge 1)         -> {2.0, 2.0, 2.0}
  const auto expected = torch::tensor({{4.0, 4.0, 4.0}, {2.0, 2.0, 2.0}});

  EXPECT_EQ(out.dim(), 2);
  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 3);
  EXPECT_TRUE(out.allclose(expected));
}

TEST(ScatterOps, EmptyInputs) {
  const auto src = torch::empty({0});
  const auto index = torch::empty({0}, torch::kLong);

  const auto out = tgn::scatter_max(src, index, 5);
  EXPECT_EQ(out.size(0), 5);
  EXPECT_TRUE(out.equal(torch::zeros({5})));
}

TEST(ScatterOps, EmptyInputs2D) {
  const auto src = torch::empty({0, 8});
  const auto index = torch::empty({0}, torch::kLong);

  const auto out = tgn::scatter_add(src, index, 5);
  EXPECT_EQ(out.dim(), 2);
  EXPECT_EQ(out.size(0), 5);
  EXPECT_EQ(out.size(1), 8);
  EXPECT_TRUE(out.equal(torch::zeros({5, 8})));
}
