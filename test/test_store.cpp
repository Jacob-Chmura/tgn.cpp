#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

#include "lib.h"

TEST(TGStoreTest, MakeStoreInit) {
  const std::int64_t n = 10;
  const std::int64_t d = 8;
  const std::int64_t m = 3;
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::full({n}, 5, torch::kLong),
      .t = torch::linspace(0, 1, n),
      .msg = torch::randn({n, d}),
      .neg_dst = torch::randint(0, 6, {n, m}, torch::kLong)};

  const auto store = tgn::make_store(opts);
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(store->num_edges(), n);
  EXPECT_EQ(store->msg_dim(), d);
  EXPECT_EQ(store->num_nodes(), 6);  // Max ID 5 + 1
}

TEST(TGStoreTest, RejectsInvalidShapes) {
  const std::int64_t n = 10;
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({5}, torch::kLong),  // Mismatching 'n'
      .t = torch::zeros({n}),
      .msg = torch::zeros({n, 4}),
      .neg_dst = torch::zeros({n}, torch::kLong)};

  EXPECT_THROW(tgn::make_store(opts), c10::Error);
}

TEST(TGStoreTest, RejectsInvalidNegativesShapes) {
  const std::int64_t n = 10;
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .t = torch::zeros({n}),
      .msg = torch::zeros({n, 4}),
      .neg_dst = torch::zeros({n}, torch::kLong)};  // Should be [n, m]

  EXPECT_THROW(tgn::make_store(opts), c10::Error);
}

TEST(TGStoreTest, RejectsOutOfRangeNegatives) {
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::tensor({0, 1}, torch::kLong),
      .dst = torch::tensor({1, 2}, torch::kLong),
      .t = torch::zeros({2}),
      .msg = torch::zeros({2, 4}),
      .neg_dst =
          torch::tensor({{99}}, torch::kLong)};  // 99 is out of range [0, 2]

  EXPECT_THROW(tgn::make_store(opts), c10::Error);
}

TEST(TGStoreTest, RejectsFloatingPointIDs) {
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::randn({10}),  // Float instead of Long
      .dst = torch::zeros({10}, torch::kLong),
      .t = torch::zeros({10}),
      .msg = torch::zeros({10, 4}),
      .neg_dst = std::nullopt};

  EXPECT_THROW(tgn::make_store(opts), c10::Error);
}

TEST(TGStoreTest, GetBatchWithoutNegatives) {
  const std::int64_t n = 100;
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::arange(n, torch::kLong),
                                  .dst = torch::zeros({n}, torch::kLong),
                                  .t = torch::zeros({n}),
                                  .msg = torch::zeros({n, 4}),
                                  .neg_dst = std::nullopt};
  const auto store = tgn::make_store(opts);

  const std::size_t start = 10;
  const std::size_t batch_size = 20;
  const auto batch = store->get_batch(start, batch_size);

  ASSERT_EQ(batch.src.size(0), batch_size);
  EXPECT_EQ(batch.src[0].item<std::int64_t>(), 10);
  EXPECT_EQ(batch.src[19].item<std::int64_t>(), 29);
  EXPECT_FALSE(batch.neg_dst.has_value());
}

TEST(TGStoreTest, GetBatchWithMultiNegatives) {
  const std::int64_t n = 100;
  const std::int64_t m = 3;

  auto negs = torch::zeros({n, m}, torch::kLong);
  for (int i = 0; i < n; ++i) {
    negs[i].fill_(i);
  }

  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::arange(n, torch::kLong),
                                  .dst = torch::full({n}, n, torch::kLong),
                                  .t = torch::zeros({n}),
                                  .msg = torch::zeros({n, 4}),
                                  .neg_dst = negs};
  const auto store = make_store(opts);

  const std::size_t start = 10;
  const std::size_t batch_size = 20;
  const auto batch =
      store->get_batch(start, batch_size, tgn::NegStrategy::PreComputed);

  ASSERT_EQ(batch.src.size(0), 20);
  ASSERT_TRUE(batch.neg_dst.has_value());
  EXPECT_EQ(batch.neg_dst->size(0), 20);
  EXPECT_EQ(batch.neg_dst->size(1), m);

  // Verify slicing: first row of batch should be row 10 of original
  EXPECT_EQ((*batch.neg_dst)[0][0].item<std::int64_t>(), 10);
  EXPECT_EQ((*batch.neg_dst)[19][0].item<std::int64_t>(), 29);
}

TEST(TGStoreTest, GetBatchPartialTail) {
  const std::int64_t n = 100;
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::arange(n, torch::kLong),
                                  .dst = torch::zeros({n}, torch::kLong),
                                  .t = torch::zeros({n}),
                                  .msg = torch::zeros({n, 4}),
                                  .neg_dst = std::nullopt};
  const auto store = tgn::make_store(opts);

  // Start near the end and request a size that exceeds total edges
  const std::size_t start = 95;
  const std::size_t batch_size = 10;
  const auto batch = store->get_batch(start, batch_size);

  ASSERT_EQ(batch.src.size(0), 5);
  EXPECT_EQ(batch.src[0].item<std::int64_t>(), 95);
  EXPECT_EQ(batch.src[4].item<std::int64_t>(), 99);

  // Start exactly at the end should return an empty batch.
  const auto empty_batch = store->get_batch(100, 10);
  EXPECT_EQ(empty_batch.src.size(0), 0);
}

TEST(TGStoreTest, GatherMsgs) {
  const std::int64_t n = 5;
  const std::int64_t d = 2;
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .t = torch::zeros({n}),
      .msg = torch::tensor(
          {{1.1, 1.1}, {2.2, 2.2}, {3.3, 3.3}, {4.4, 4.4}, {5.5, 5.5}}),
      .neg_dst = std::nullopt};
  const auto store = tgn::make_store(opts);

  const auto e_ids = torch::tensor({0, 4, 1}, torch::kLong);
  const auto msgs = store->gather_msgs(e_ids);

  ASSERT_EQ(msgs.dim(), 2);
  ASSERT_EQ(msgs.size(0), 3);
  ASSERT_EQ(msgs.size(1), d);
  EXPECT_FLOAT_EQ(msgs[0][0].item<float>(), 1.1F);
  EXPECT_FLOAT_EQ(msgs[1][0].item<float>(), 5.5F);
  EXPECT_FLOAT_EQ(msgs[2][0].item<float>(), 2.2F);
}

TEST(TGStoreTest, GatherTimestamps) {
  const std::int64_t n = 5;
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .t = torch::tensor({10.1, 20.2, 30.3, 40.4, 50.5}),
      .msg = torch::zeros({n, 4}),
      .neg_dst = std::nullopt};
  const auto store = tgn::make_store(opts);

  const auto e_ids = torch::tensor({4, 0, 2}, torch::kLong);
  const auto timestamps = store->gather_timestamps(e_ids);

  ASSERT_EQ(timestamps.dim(), 1);
  ASSERT_EQ(timestamps.size(0), 3);
  EXPECT_FLOAT_EQ(timestamps[0].item<float>(), 50.5F);
  EXPECT_FLOAT_EQ(timestamps[1].item<float>(), 10.1F);
  EXPECT_FLOAT_EQ(timestamps[2].item<float>(), 30.3F);
}

TEST(TGStoreTest, HandlesEmptyInputs) {
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::empty({0}, torch::kLong),
                                  .dst = torch::empty({0}, torch::kLong),
                                  .t = torch::empty({0}),
                                  .msg = torch::empty({0, 4}),
                                  .neg_dst = std::nullopt};

  const auto store = tgn::make_store(opts);
  EXPECT_EQ(store->num_edges(), 0);
  EXPECT_EQ(store->num_nodes(), 0);
}
