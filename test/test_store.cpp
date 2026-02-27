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

TEST(TGStoreTest, GetBatchNegStrategyNone) {
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

TEST(TGStoreTest, GetBatchNegStrategyPreComputed) {
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

TEST(TGStoreTest, GetBatchNegStrategyPreComputedThrowsIfNull) {
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::zeros({10}, torch::kLong),
                                  .dst = torch::zeros({10}, torch::kLong),
                                  .t = torch::zeros({10}),
                                  .msg = torch::zeros({10, 1}),
                                  .neg_dst = std::nullopt};

  const auto store = tgn::make_store(opts);

  // Should throw because strategy is PreComputed but neg_dst is missing
  EXPECT_THROW(store->get_batch(0, 5, tgn::NegStrategy::PreComputed),
               c10::Error);
}

TEST(TGStoreTest, GetBatchNegStrategyRandom) {
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::tensor({0, 1}, torch::kLong),
      .dst = torch::tensor({10, 20}, torch::kLong),  // IDs in train are 10, 20
      .t = torch::zeros({2}),
      .msg = torch::zeros({2, 1}),
      .val_start = 2};

  const auto store = tgn::make_store(opts);
  const auto batch = store->get_batch(0, 10, tgn::NegStrategy::Random);

  ASSERT_TRUE(batch.neg_dst.has_value());

  // Sampler should stay within [min_id, max_id] of training destinations
  const auto neg_data = batch.neg_dst->flatten();
  for (int i = 0; i < neg_data.size(0); ++i) {
    const auto val = neg_data[i].item<std::int64_t>();
    EXPECT_GE(val, 10);
    EXPECT_LE(val, 20);
  }
}

TEST(TGStoreTest, GetBatchNegStrategyRandomThrowsIfTrainEmpty) {
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::zeros({10}, torch::kLong),
                                  .dst = torch::zeros({10}, torch::kLong),
                                  .t = torch::zeros({10}),
                                  .msg = torch::zeros({10, 1}),
                                  .val_start = 0};

  const auto store = tgn::make_store(opts);

  // TODO(kuba): There might be a use case for factoring out negatives
  // into a more general API (e.g. you want random negatives in validation?)
  EXPECT_THROW(store->get_batch(0, 5, tgn::NegStrategy::Random), c10::Error);
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

TEST(TGStoreTest, SplitsWithCustomBoundaries) {
  const std::int64_t n = 10;
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::zeros({n}, torch::kLong),
                                  .dst = torch::zeros({n}, torch::kLong),
                                  .t = torch::zeros({n}),
                                  .msg = torch::zeros({n, 1}),
                                  .val_start = 6,
                                  .test_start = 8};

  const auto store = tgn::make_store(opts);

  EXPECT_EQ(store->train_split().start(), 0);
  EXPECT_EQ(store->train_split().end(), 6);

  EXPECT_EQ(store->val_split().start(), 6);
  EXPECT_EQ(store->val_split().end(), 8);

  EXPECT_EQ(store->test_split().start(), 8);
  EXPECT_EQ(store->test_split().end(), 10);
}

TEST(TGStoreTest, SplitsWithNoBoundaries) {
  const std::int64_t n = 10;
  const auto opts =
      tgn::InMemoryTGStoreOptions{.src = torch::zeros({n}, torch::kLong),
                                  .dst = torch::zeros({n}, torch::kLong),
                                  .t = torch::zeros({n}),
                                  .msg = torch::zeros({n, 1})};

  const auto store = tgn::make_store(opts);

  // Default behavior should put everything in train
  EXPECT_EQ(store->train_split().size(), 10);
  EXPECT_EQ(store->val_split().size(), 0);
  EXPECT_EQ(store->test_split().size(), 0);
}

TEST(TGStoreTest, LabelSplitEmpty) {
  const auto s = tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                                  .dst = torch::zeros({5}, torch::kLong),
                                  .t = torch::arange(5),
                                  .msg = torch::zeros({5, 1})});
  EXPECT_EQ(s->train_label_split().size(), 0);
  EXPECT_EQ(s->val_label_split().size(), 0);
  EXPECT_EQ(s->test_label_split().size(), 0);
}

TEST(TGStoreTest, LabelSplitThreeDistinct) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::zeros({3}, torch::kLong),
                       .label_t = torch::tensor({15, 25, 45}, torch::kLong),
                       .label_y_true = torch::zeros({3})});
  EXPECT_EQ(s->train_label_split().size(), 3);
  EXPECT_EQ(s->val_label_split().size(), 0);
  EXPECT_EQ(s->test_label_split().size(), 0);
}

TEST(TGStoreTest, LabelSplitThreeGrouped) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::zeros({3}, torch::kLong),
                       .label_t = torch::tensor({15, 15, 25}, torch::kLong),
                       .label_y_true = torch::zeros({3})});
  EXPECT_EQ(s->train_label_split().size(), 2);  // 2 unique timestamps
  EXPECT_EQ(s->val_label_split().size(), 0);
  EXPECT_EQ(s->test_label_split().size(), 0);
}

TEST(TGStoreTest, LabelSplitSingle) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::zeros({1}, torch::kLong),
                       .label_t = torch::tensor({15}, torch::kLong),
                       .label_y_true = torch::zeros({1})});
  EXPECT_EQ(s->train_label_split().size(), 1);
  EXPECT_EQ(s->val_label_split().size(), 0);
  EXPECT_EQ(s->test_label_split().size(), 0);
}

TEST(TGStoreTest, LabelSplitsWithCustomBoundaries) {
  // Custom split: Train [0,6), Val [6,8), Test [8,10)
  // Train ends at edge index 6 (t=70)
  // Val ends at edge index 8 (t=90)
  const auto opts = tgn::InMemoryTGStoreOptions{
      .src = torch::zeros({10}, torch::kLong),
      .dst = torch::zeros({10}, torch::kLong),
      .t = torch::tensor({10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
                         torch::kLong),
      .msg = torch::zeros({10, 1}),
      .val_start = 6,
      .test_start = 8,
      // Labels:
      // t=15 (Index 0) -> Train (15 < 70)
      // t=75 (Index 1) -> Val   (70 <= 75 < 90)
      // t=95 (Index 2) -> Test  (95 >= 90)
      .label_n_id = torch::tensor({1, 2, 3}, torch::kLong),
      .label_t = torch::tensor({15, 75, 95}, torch::kLong),
      .label_y_true = torch::zeros({3})};

  const auto store = tgn::make_store(opts);

  EXPECT_EQ(store->train_label_split().start(), 0);
  EXPECT_EQ(store->train_label_split().end(), 1);

  EXPECT_EQ(store->val_label_split().start(), 1);
  EXPECT_EQ(store->val_label_split().end(), 2);

  EXPECT_EQ(store->test_label_split().start(), 2);
  EXPECT_EQ(store->test_label_split().end(), 3);
}

TEST(TGStoreTest, GetStopEIdEmptyThrows) {
  const auto s = tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                                  .dst = torch::zeros({5}, torch::kLong),
                                  .t = torch::arange(5),
                                  .msg = torch::zeros({5, 1})});
  EXPECT_THROW(s->get_stop_e_id_for_label_event(0), std::out_of_range);
}

TEST(TGStoreTest, GetStopEIdThreeDistinct) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::zeros({3}, torch::kLong),
                       .label_t = torch::tensor({15, 25, 45}, torch::kLong),
                       .label_y_true = torch::zeros({3})});
  EXPECT_EQ(s->get_stop_e_id_for_label_event(0), 1);  // Before t=20
  EXPECT_EQ(s->get_stop_e_id_for_label_event(1), 2);  // Before t=30
  EXPECT_EQ(s->get_stop_e_id_for_label_event(2), 4);  // Before t=50
}

TEST(TGStoreTest, GetStopEIdThreeGrouped) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::zeros({3}, torch::kLong),
                       .label_t = torch::tensor({15, 15, 25}, torch::kLong),
                       .label_y_true = torch::zeros({3})});

  // Stop ID for the first group (t=15)
  EXPECT_EQ(s->get_stop_e_id_for_label_event(0), 1);

  // Stop ID for the second group (t=25)
  EXPECT_EQ(s->get_stop_e_id_for_label_event(1), 2);
}

TEST(TGStoreTest, GetStopEIdSingle) {
  const auto s = tgn::make_store({.src = torch::zeros({2}, torch::kLong),
                                  .dst = torch::zeros({2}, torch::kLong),
                                  .t = torch::tensor({10, 20}, torch::kLong),
                                  .msg = torch::zeros({2, 1}),
                                  .label_n_id = torch::zeros({1}, torch::kLong),
                                  .label_t = torch::tensor({25}, torch::kLong),
                                  .label_y_true = torch::zeros({1})});
  EXPECT_EQ(s->get_stop_e_id_for_label_event(0), 2);  // Matches end of edges
}
TEST(TGStoreTest, GetLabelEventEmptyThrows) {
  const auto s = tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                                  .dst = torch::zeros({5}, torch::kLong),
                                  .t = torch::arange(5),
                                  .msg = torch::zeros({5, 1})});
  EXPECT_THROW(s->get_label_event(0), std::out_of_range);
}

TEST(TGStoreTest, GetLabelEventThreeDistinct) {
  const auto s = tgn::make_store(
      {.src = torch::zeros({5}, torch::kLong),
       .dst = torch::zeros({5}, torch::kLong),
       .t = torch::arange(5),
       .msg = torch::zeros({5, 1}),
       .label_n_id = torch::tensor({100, 200, 300}, torch::kLong),
       .label_t = torch::tensor({1, 2, 3}, torch::kLong),
       .label_y_true = torch::zeros({3})});
  EXPECT_EQ(s->get_label_event(0).n_id[0].item<int64_t>(), 100);
  EXPECT_EQ(s->get_label_event(1).n_id[0].item<int64_t>(), 200);
  EXPECT_EQ(s->get_label_event(2).n_id[0].item<int64_t>(), 300);

  EXPECT_EQ(s->get_label_event(0).y_true[0].item<float>(), 0);
  EXPECT_EQ(s->get_label_event(1).y_true[0].item<float>(), 0);
  EXPECT_EQ(s->get_label_event(2).y_true[0].item<float>(), 0);
}

TEST(TGStoreTest, GetLabelEventThreeGrouped) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::arange(5),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::tensor({1, 2, 3}, torch::kLong),
                       .label_t = torch::tensor({10, 10, 20}, torch::kLong),
                       .label_y_true = torch::zeros({3})});
  auto event = s->get_label_event(0);
  EXPECT_EQ(event.n_id.size(0), 2);  // Groups node 1 and 2
  EXPECT_EQ(event.n_id[0].item<int64_t>(), 1);
  EXPECT_EQ(event.n_id[1].item<int64_t>(), 2);

  event = s->get_label_event(1);
  EXPECT_EQ(event.n_id.size(0), 1);  // Groups node 1 and 2
  EXPECT_EQ(event.n_id[0].item<int64_t>(), 3);
}

TEST(TGStoreTest, GetLabelEventSingle) {
  const auto s =
      tgn::make_store({.src = torch::zeros({5}, torch::kLong),
                       .dst = torch::zeros({5}, torch::kLong),
                       .t = torch::arange(5),
                       .msg = torch::zeros({5, 1}),
                       .label_n_id = torch::tensor({999}, torch::kLong),
                       .label_t = torch::tensor({10}, torch::kLong),
                       .label_y_true = torch::zeros({1})});
  EXPECT_EQ(s->get_label_event(0).n_id[0].item<int64_t>(), 999);
  EXPECT_EQ(s->get_label_event(0).y_true[0].item<float>(), 0);
}

TEST(TGRange, ValidRange) {
  const auto split = tgn::Range(1, 5);
  EXPECT_EQ(split.start(), 1);
  EXPECT_EQ(split.end(), 5);
  EXPECT_EQ(split.size(), 4);
}

TEST(TGRange, ValidEmptyRange) {
  const auto split = tgn::Range();
  EXPECT_EQ(split.start(), 0);
  EXPECT_EQ(split.end(), 0);
  EXPECT_EQ(split.size(), 0);
}

TEST(TGRange, RejectsInvalidRange) {
  EXPECT_THROW(tgn::Range(5, 3), std::out_of_range);
}
