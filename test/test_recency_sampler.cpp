#include <gtest/gtest.h>
#include <torch/torch.h>

#include "recency_sampler.h"

TEST(RecencySampler, CallOnEmpty) {
  auto loader = tgn::LastNeighborLoader(10, 100);
  const auto [unique_id, edge_index, e_id] =
      loader(torch::tensor({2, 2, 1, 3}));

  // Should return the unique IDs provided (sorted), but empty edges
  EXPECT_TRUE(unique_id.equal(torch::tensor({1, 2, 3}, torch::kLong)));
  EXPECT_TRUE(edge_index.equal(torch::empty({2, 0}, torch::kLong)));
  EXPECT_TRUE(e_id.equal(torch::empty({0}, torch::kLong)));
}

TEST(RecencySampler, InsertThenCall) {
  auto loader = tgn::LastNeighborLoader(1, 10);

  // Insert edge: 1 -> 2
  loader.insert(torch::tensor({1}), torch::tensor({2}));

  // Sample node 2. It should have node 1 as a neighbor.
  const auto [unique_id, edge_index, e_id] = loader(torch::tensor({2}));

  // unique_id should contain {1, 2} (sorted)
  EXPECT_TRUE(unique_id.equal(torch::tensor({1, 2}, torch::kLong)));

  // edge_index check: stack({local_nbr, local_node})
  // Node 1 is unique_id[0], Node 2 is unique_id[1].
  // Edge is 1 (nbr) -> 2 (target), so local indices are [0, 1]
  EXPECT_TRUE(edge_index.equal(torch::tensor({{0}, {1}}, torch::kLong)));
  EXPECT_TRUE(e_id.equal(torch::tensor({0}, torch::kLong)));
}

TEST(RecencySampler, InsertNumInteractionsExceedsBufferSize) {
  auto loader = tgn::LastNeighborLoader(2, 10);  // buffer size is 2

  // Edge 0: 1-2, Edge 1: 1-3, Edge 2: 1-4
  loader.insert(torch::tensor({1}), torch::tensor({2}));
  loader.insert(torch::tensor({1}), torch::tensor({3}));
  loader.insert(torch::tensor({1}), torch::tensor({4}));

  const auto [unique_id, edge_index, e_id] = loader(torch::tensor({1}));

  // Should only have the 2 most recent: (1-4) and (1-3).
  // Note: topk returns indices in order of values, so [2, 1]
  EXPECT_TRUE(e_id.equal(torch::tensor({2, 1}, torch::kLong)));
  EXPECT_TRUE(unique_id.equal(torch::tensor({1, 3, 4}, torch::kLong)));

  // Edge 2 is 4->1 => local [unique_id.index(4), unique_id.index(1)] = [2, 0]
  // Edge 1 is 3->1 => local [unique_id.index(3), unique_id.index(1)] = [1, 0]
  EXPECT_TRUE(edge_index.equal(torch::tensor({{2, 1}, {0, 0}}, torch::kLong)));
}

TEST(RecencySampler, BiDirectionalCheck) {
  auto loader = tgn::LastNeighborLoader(5, 10);
  loader.insert(torch::tensor({1}), torch::tensor({2}));

  const auto [u1, ei1, e1] = loader(torch::tensor({1}));
  const auto [u2, ei2, e2] = loader(torch::tensor({2}));

  // Both nodes should report the same edge ID 0
  EXPECT_TRUE(e1.equal(torch::tensor({0}, torch::kLong)));
  EXPECT_TRUE(e2.equal(torch::tensor({0}, torch::kLong)));

  // Node 1's neighbor is 2; Node 2's neighbor is 1
  EXPECT_TRUE(u1.equal(torch::tensor({1, 2}, torch::kLong)));
  EXPECT_TRUE(u2.equal(torch::tensor({1, 2}, torch::kLong)));

  EXPECT_TRUE(ei1.equal(torch::tensor({{1}, {0}}, torch::kLong)));
  EXPECT_TRUE(ei2.equal(torch::tensor({{0}, {1}}, torch::kLong)));
}

TEST(RecencySampler, ResetThenCall) {
  auto loader = tgn::LastNeighborLoader(5, 10);
  loader.insert(torch::tensor({1}), torch::tensor({2}));
  loader.reset_state();

  const auto [unique_id, edge_index, e_id] = loader(torch::tensor({1}));

  EXPECT_TRUE(unique_id.equal(torch::tensor({1}, torch::kLong)));
  EXPECT_TRUE(edge_index.equal(torch::empty({2, 0}, torch::kLong)));
  EXPECT_TRUE(e_id.equal(torch::empty({0}, torch::kLong)));
}

TEST(RecencySampler, LargeNodeIDRange) {
  auto loader = tgn::LastNeighborLoader(2, 1000);

  loader.insert(torch::tensor({10}), torch::tensor({999}));
  const auto [unique_id, edge_index, e_id] = loader(torch::tensor({999}));

  EXPECT_TRUE(unique_id.equal(torch::tensor({10, 999}, torch::kLong)));
  EXPECT_TRUE(edge_index.equal(torch::tensor({{0}, {1}}, torch::kLong)));
  EXPECT_TRUE(e_id.equal(torch::tensor({0}, torch::kLong)));
}
