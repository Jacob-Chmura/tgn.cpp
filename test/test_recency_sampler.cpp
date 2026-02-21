#include <gtest/gtest.h>
#include <torch/torch.h>

#include "recency_sampler.h"

TEST(RecencySampler, Init) {
  auto loader = tgn::LastNeighborLoader(5, 100);
  EXPECT_EQ(loader.buffer_size_, 5);
  EXPECT_EQ(loader.cur_e_id_, 0);
  // Ensure buffers are initialized with -1
  EXPECT_TRUE(loader.buffer_e_id_.eq(-1).all().item<bool>());
}

TEST(RecencySampler, CallOnEmpty) {
  auto loader = tgn::LastNeighborLoader(10, 100);
  const auto ids = torch::tensor({1, 2, 3}, torch::kLong);

  const auto [unique_id, edge_index, e_id] = loader(ids);

  // Should return the unique IDs provided, but empty edges
  EXPECT_TRUE(unique_id.equal(ids));
  EXPECT_EQ(edge_index.size(1), 0);
  EXPECT_EQ(e_id.numel(), 0);
}

TEST(RecencySampler, InsertAndSample) {
  // 1 nbr per node, 10 nodes total
  auto loader = tgn::LastNeighborLoader(1, 10);

  // Insert edge: 1 -> 2
  const auto src = torch::tensor({1}, torch::kLong);
  const auto dst = torch::tensor({2}, torch::kLong);
  loader.insert(src, dst);

  // Sample node 2. It should have node 1 as a neighbor.
  const auto [unique_id, edge_index, e_id] =
      loader(torch::tensor({2}, torch::kLong));

  // unique_id should contain both {1, 2} because 1 is a neighbor of 2
  EXPECT_EQ(unique_id.size(0), 2);
  // edge_index should represent 1 -> 2
  // Since it's stack({local_nbr, local_node}), we check the shape
  EXPECT_EQ(edge_index.size(1), 1);
  EXPECT_EQ(e_id.item<int64_t>(), 0);  // First edge ID is 0
}

TEST(RecencySampler, InsertNumInteractionsExceedsBufferSize) {
  // Buffer size of 2. We will insert 3 interactions for the same node.
  auto loader = tgn::LastNeighborLoader(2, 10);

  // Interaction 0: 1-2
  // Interaction 1: 1-3
  // Interaction 2: 1-4
  loader.insert(torch::tensor({1}), torch::tensor({2}));
  loader.insert(torch::tensor({1}), torch::tensor({3}));
  loader.insert(torch::tensor({1}), torch::tensor({4}));

  const auto [unique_id, edge_index, e_id] =
      loader(torch::tensor({1}, torch::kLong));

  // Should only have the 2 most recent: (1-3) and (1-4)
  // Edge IDs should be 1 and 2 (0 was evicted)
  EXPECT_EQ(e_id.size(0), 2);
  const auto e_id_list = e_id.view({-1});
  EXPECT_TRUE(e_id_list.equal(torch::tensor({2, 1}, torch::kLong)));
}

TEST(RecencySampler, BiDirectionalCheck) {
  auto loader = tgn::LastNeighborLoader(5, 10);
  loader.insert(torch::tensor({1}), torch::tensor({2}));

  // Check node 1
  const auto [u1, ei1, e1] = loader(torch::tensor({1}, torch::kLong));
  // Check node 2
  const auto [u2, ei2, e2] = loader(torch::tensor({2}, torch::kLong));

  // Both should find each other
  EXPECT_EQ(e1.numel(), 1);
  EXPECT_EQ(e2.numel(), 1);
  EXPECT_EQ(e1.item<int64_t>(), e2.item<int64_t>());
}

TEST(RecencySampler, Reset) {
  auto loader = tgn::LastNeighborLoader(5, 10);
  loader.insert(torch::tensor({1}), torch::tensor({2}));
  loader.reset_state();

  EXPECT_EQ(loader.cur_e_id_, 0);
  const auto [unique_id, edge_index, e_id] =
      loader(torch::tensor({1}, torch::kLong));
  EXPECT_EQ(e_id.numel(), 0);  // Should be empty again
}

TEST(RecencySampler, LargeNodeIDRange) {
  // Test that our assoc_ logic handles sparse/large global IDs within num_nodes
  auto loader = tgn::LastNeighborLoader(2, 1000);
  const auto src = torch::tensor({10}, torch::kLong);
  const auto dst = torch::tensor({999}, torch::kLong);  // Boundary test

  loader.insert(src, dst);
  const auto [unique_id, edge_index, e_id] =
      loader(torch::tensor({999}, torch::kLong));

  EXPECT_EQ(e_id.numel(), 1);
}
