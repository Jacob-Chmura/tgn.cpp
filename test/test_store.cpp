#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "lib.h"

TEST(TGStoreTest, MakeStoreInitialization) {
  const std::int64_t n = 10, d = 8;
  const auto opts =
      InMemoryTGStoreOptions{.src = torch::zeros({n}, torch::kLong),
                             .dst = torch::full({n}, 5, torch::kLong),
                             .t = torch::linspace(0, 1, n),
                             .msg = torch::randn({n, d}),
                             .neg_dst = torch::zeros({n}, torch::kLong)};

  const auto store = make_store(opts);
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(store->num_edges(), n);
  EXPECT_EQ(store->msg_dim(), d);
  EXPECT_EQ(store->num_nodes(), 6);  // Max ID 5 + 1
}

TEST(TGStoreTest, RejectsInvalidShapes) {
  const std::int64_t n = 10;
  const auto opts = InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({5}, torch::kLong),  // Mismatching 'n'
      .t = torch::zeros({n}),
      .msg = torch::zeros({n, 4}),
      .neg_dst = torch::zeros({n}, torch::kLong)};

  EXPECT_THROW(make_store(opts), c10::Error);
}

TEST(TGStoreTest, RejectsFloatingPointIDs) {
  const auto opts = InMemoryTGStoreOptions{
      .src = torch::randn({10}),  // Float instead of Long
      .dst = torch::zeros({10}, torch::kLong),
      .t = torch::zeros({10}),
      .msg = torch::zeros({10, 4}),
      .neg_dst = torch::zeros({10}, torch::kLong)};

  EXPECT_THROW(make_store(opts), c10::Error);
}

TEST(TGStoreTest, GetBatch) {
  const std::int64_t n = 100;
  const auto opts =
      InMemoryTGStoreOptions{.src = torch::arange(n, torch::kLong),
                             .dst = torch::zeros({n}, torch::kLong),
                             .t = torch::zeros({n}),
                             .msg = torch::zeros({n, 4}),
                             .neg_dst = torch::zeros({n}, torch::kLong)};
  const auto store = make_store(opts);

  const std::size_t start = 10;
  const std::size_t batch_size = 20;
  const auto batch = store->get_batch(start, batch_size);

  ASSERT_EQ(batch.src.size(0), batch_size);
  EXPECT_EQ(batch.src[0].item<std::int64_t>(), 10);
  EXPECT_EQ(batch.src[19].item<std::int64_t>(), 29);
}

TEST(TGStoreTest, GetBatchPartialTail) {
  const std::int64_t n = 100;
  const auto opts =
      InMemoryTGStoreOptions{.src = torch::arange(n, torch::kLong),
                             .dst = torch::zeros({n}, torch::kLong),
                             .t = torch::zeros({n}),
                             .msg = torch::zeros({n, 4}),
                             .neg_dst = torch::zeros({n}, torch::kLong)};
  const auto store = make_store(opts);

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
  const auto opts = InMemoryTGStoreOptions{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .t = torch::zeros({n}),
      .msg = torch::tensor(
          {{1.1, 1.1}, {2.2, 2.2}, {3.3, 3.3}, {4.4, 4.4}, {5.5, 5.5}}),
      .neg_dst = torch::zeros({n}, torch::kLong)};
  const auto store = make_store(opts);

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
  const auto opts =
      InMemoryTGStoreOptions{.src = torch::zeros({n}, torch::kLong),
                             .dst = torch::zeros({n}, torch::kLong),
                             .t = torch::tensor({10.1, 20.2, 30.3, 40.4, 50.5}),
                             .msg = torch::zeros({n, 4}),
                             .neg_dst = torch::zeros({n}, torch::kLong)};
  const auto store = make_store(opts);

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
      InMemoryTGStoreOptions{.src = torch::empty({0}, torch::kLong),
                             .dst = torch::empty({0}, torch::kLong),
                             .t = torch::empty({0}),
                             .msg = torch::empty({0, 4}),
                             .neg_dst = torch::empty({0}, torch::kLong)};

  const auto store = make_store(opts);
  EXPECT_EQ(store->num_edges(), 0);
  EXPECT_EQ(store->num_nodes(), 0);
}
