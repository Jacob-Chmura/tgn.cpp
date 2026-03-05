#include <gtest/gtest.h>
#include <torch/types.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "tgn.h"

struct TestData {
  torch::Tensor src, dst, time, msg;
  std::optional<torch::Tensor> neg_dst;
  std::optional<torch::Tensor> label_n_id, label_time, label_target;
  std::optional<std::size_t> val_start, test_start;
};

class TGStoreFixture : public ::testing::Test {
 public:
  virtual ~TGStoreFixture() = default;
  virtual auto make_store(TestData data) -> std::shared_ptr<tgn::TGStore> = 0;
};

class InMemoryTGStoreFixture : public TGStoreFixture {
 public:
  auto make_store(TestData data) -> std::shared_ptr<tgn::TGStore> override {
    return tgn::TGStore::from_memory(tgn::Batch{.src = data.src,
                                                .dst = data.dst,
                                                .time = data.time,
                                                .msg = data.msg,
                                                .neg_dst = data.neg_dst},
                                     data.label_n_id, data.label_time,
                                     data.label_target, data.val_start,
                                     data.test_start);
  }
};

class TgufTGStoreFixture : public TGStoreFixture {
 public:
  TgufTGStoreFixture() {
    tguf_path_ = std::filesystem::temp_directory_path() /
                 (std::to_string(std::random_device{}()) + ".tguf");
  }

  ~TgufTGStoreFixture() override {
    if (std::filesystem::exists(tguf_path_)) {
      std::filesystem::remove(tguf_path_);
    }
  }

  auto make_store(TestData data) -> std::shared_ptr<tgn::TGStore> override {
    return make_store_with_opts(std::move(data));
  }

  auto make_store_with_opts(
      TestData data,
      std::optional<std::size_t> val_start_override = std::nullopt,
      std::optional<std::size_t> test_start_override = std::nullopt)
      -> std::shared_ptr<tgn::TGStore> {
    const tgn::TGUFSchema schema{
        .path = tguf_path_.string(),
        .edge_capacity = static_cast<std::size_t>(data.src.size(0)),
        .label_capacity =
            data.label_n_id.has_value()
                ? static_cast<std::size_t>(data.label_n_id->size(0))
                : 0,
        .msg_dim = static_cast<std::size_t>(data.msg.size(1)),
        .label_dim = data.label_target.has_value()
                         ? static_cast<std::size_t>(data.label_target->size(1))
                         : 0,
        .negatives_per_edge =
            data.neg_dst.has_value()
                ? static_cast<std::size_t>(data.neg_dst->size(1))
                : 0,
        .val_start = data.val_start,
        .test_start = data.test_start,
    };
    tgn::TGUFBuilder builder(schema);

    builder.append_edges(tgn::Batch{.src = data.src,
                                    .dst = data.dst,
                                    .time = data.time,
                                    .msg = data.msg,
                                    .neg_dst = data.neg_dst});
    if (data.label_n_id.has_value()) {
      builder.append_labels(*data.label_n_id, *data.label_time,
                            *data.label_target);
    }
    builder.finalize();
    const auto val_start =
        val_start_override.has_value() ? val_start_override : data.val_start;
    const auto test_start =
        test_start_override.has_value() ? test_start_override : data.test_start;
    return tgn::TGStore::from_tguf(tguf_path_, val_start, test_start);
  }

 private:
  std::filesystem::path tguf_path_;
};

template <typename T>
class TGStoreTest : public T {};

using StoreTypes = ::testing::Types<InMemoryTGStoreFixture, TgufTGStoreFixture>;
TYPED_TEST_SUITE(TGStoreTest, StoreTypes);

TYPED_TEST(TGStoreTest, MakeStoreInit) {
  const std::int64_t n = 10;
  const std::int64_t d = 8;
  const std::int64_t m = 3;
  const auto store = this->make_store(
      TestData{.src = torch::zeros({n}, torch::kLong),
               .dst = torch::full({n}, 5, torch::kLong),
               .time = torch::arange(n, torch::kLong),
               .msg = torch::randn({n, d}),
               .neg_dst = torch::randint(0, 6, {n, m}, torch::kLong)});
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(store->edge_count(), n);
  EXPECT_EQ(store->msg_dim(), d);
  EXPECT_EQ(store->label_dim(), 0);
  EXPECT_EQ(store->node_count(), 6);  // Max ID 5 + 1
}

TYPED_TEST(TGStoreTest, GetBatchNegStrategyNone) {
  const std::int64_t n = 100;
  const auto store =
      this->make_store(TestData{.src = torch::arange(n, torch::kLong),
                                .dst = torch::zeros({n}, torch::kLong),
                                .time = torch::zeros({n}, torch::kLong),
                                .msg = torch::zeros({n, 4}),
                                .neg_dst = std::nullopt});

  const std::size_t start = 10;
  const std::size_t batch_size = 20;
  const auto batch = store->get_batch(start, batch_size);

  ASSERT_EQ(batch.src.size(0), batch_size);
  EXPECT_EQ(batch.src[0].template item<std::int64_t>(), 10);
  EXPECT_EQ(batch.src[19].template item<std::int64_t>(), 29);
  EXPECT_FALSE(batch.neg_dst.has_value());
}

TYPED_TEST(TGStoreTest, GetBatchNegStrategyPreComputed) {
  const std::int64_t n = 100;
  const std::int64_t m = 3;

  auto negs = torch::zeros({n, m}, torch::kLong);
  for (int i = 0; i < n; ++i) {
    negs[i].fill_(i);
  }

  const auto store =
      this->make_store(TestData{.src = torch::arange(n, torch::kLong),
                                .dst = torch::full({n}, n, torch::kLong),
                                .time = torch::zeros({n}, torch::kLong),
                                .msg = torch::zeros({n, 4}),
                                .neg_dst = negs});

  const std::size_t start = 10;
  const std::size_t batch_size = 20;
  const auto batch = store->get_batch(start, batch_size,
                                      tgn::TGStore::NegStrategy::PreComputed);

  ASSERT_EQ(batch.src.size(0), 20);
  ASSERT_TRUE(batch.neg_dst.has_value());
  EXPECT_EQ(batch.neg_dst->size(0), 20);
  EXPECT_EQ(batch.neg_dst->size(1), m);

  // Verify slicing: first row of batch should be row 10 of original
  EXPECT_EQ((*batch.neg_dst)[0][0].template item<std::int64_t>(), 10);
  EXPECT_EQ((*batch.neg_dst)[19][0].template item<std::int64_t>(), 29);
}

TYPED_TEST(TGStoreTest, GetBatchNegStrategyPreComputedThrowsIfNull) {
  const auto store =
      this->make_store(TestData{.src = torch::zeros({10}, torch::kLong),
                                .dst = torch::zeros({10}, torch::kLong),
                                .time = torch::zeros({10}, torch::kLong),
                                .msg = torch::zeros({10, 1}),
                                .neg_dst = std::nullopt});

  // Should throw because strategy is PreComputed but neg_dst is missing
  EXPECT_THROW(store->get_batch(0, 5, tgn::TGStore::NegStrategy::PreComputed),
               c10::Error);
}

TYPED_TEST(TGStoreTest, GetBatchNegStrategyRandom) {
  const auto store = this->make_store(TestData{
      .src = torch::tensor({0, 1}, torch::kLong),
      .dst = torch::tensor({10, 20}, torch::kLong),  // IDs in train are 10,
      .time = torch::zeros({2}, torch::kLong),
      .msg = torch::zeros({2, 1}),
      .val_start = 2});

  const auto batch = store->get_batch(0, 10, tgn::TGStore::NegStrategy::Random);

  ASSERT_TRUE(batch.neg_dst.has_value());

  // Sampler should stay within [min_id, max_id] of training destinations
  const auto neg_data = batch.neg_dst->flatten();
  for (int i = 0; i < neg_data.size(0); ++i) {
    const auto val = neg_data[i].template item<std::int64_t>();
    EXPECT_GE(val, 10);
    EXPECT_LE(val, 20);
  }
}

TYPED_TEST(TGStoreTest, GetBatchNegStrategyRandomThrowsIfTrainEmpty) {
  const auto store =
      this->make_store(TestData{.src = torch::zeros({10}, torch::kLong),
                                .dst = torch::zeros({10}, torch::kLong),
                                .time = torch::zeros({10}, torch::kLong),
                                .msg = torch::zeros({10, 1}),
                                .val_start = 0});

  // TODO(kuba): There might be a use case for factoring out negatives
  // into a more general API (e.g. you want random negatives in validation?)
  EXPECT_THROW(store->get_batch(0, 5, tgn::TGStore::NegStrategy::Random),
               c10::Error);
}

TYPED_TEST(TGStoreTest, GetBatchPartialTail) {
  const std::int64_t n = 100;
  const auto store =
      this->make_store(TestData{.src = torch::arange(n, torch::kLong),
                                .dst = torch::zeros({n}, torch::kLong),
                                .time = torch::zeros({n}, torch::kLong),
                                .msg = torch::zeros({n, 4}),
                                .neg_dst = std::nullopt});

  // Start near the end and request a size that exceeds total edges
  const std::size_t start = 95;
  const std::size_t batch_size = 10;
  const auto batch = store->get_batch(start, batch_size);

  ASSERT_EQ(batch.src.size(0), 5);
  EXPECT_EQ(batch.src[0].template item<std::int64_t>(), 95);
  EXPECT_EQ(batch.src[4].template item<std::int64_t>(), 99);

  // Start exactly at the end should return an empty batch.
  const auto empty_batch = store->get_batch(100, 10);
  EXPECT_EQ(empty_batch.src.size(0), 0);
}

TYPED_TEST(TGStoreTest, GatherMsgs) {
  const std::int64_t n = 5;
  const std::int64_t d = 2;
  const auto store = this->make_store(TestData{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .time = torch::zeros({n}, torch::kLong),
      .msg = torch::tensor(
          {{1.1, 1.1}, {2.2, 2.2}, {3.3, 3.3}, {4.4, 4.4}, {5.5, 5.5}}),
      .neg_dst = std::nullopt});

  const auto e_ids = torch::tensor({0, 4, 1}, torch::kLong);
  const auto msgs = store->gather_msgs(e_ids);

  ASSERT_EQ(msgs.dim(), 2);
  ASSERT_EQ(msgs.size(0), 3);
  ASSERT_EQ(msgs.size(1), d);
  EXPECT_FLOAT_EQ(msgs[0][0].template item<float>(), 1.1F);
  EXPECT_FLOAT_EQ(msgs[1][0].template item<float>(), 5.5F);
  EXPECT_FLOAT_EQ(msgs[2][0].template item<float>(), 2.2F);
}

TYPED_TEST(TGStoreTest, GatherTimestamps) {
  const std::int64_t n = 5;
  const auto store = this->make_store(
      TestData{.src = torch::zeros({n}, torch::kLong),
               .dst = torch::zeros({n}, torch::kLong),
               .time = torch::tensor({101, 202, 303, 404, 505}, torch::kLong),
               .msg = torch::zeros({n, 4}),
               .neg_dst = std::nullopt});

  const auto e_ids = torch::tensor({4, 0, 2}, torch::kLong);
  const auto timestamps = store->gather_timestamps(e_ids);

  ASSERT_EQ(timestamps.dim(), 1);
  ASSERT_EQ(timestamps.size(0), 3);
  EXPECT_EQ(timestamps[0].template item<float>(), 505);
  EXPECT_EQ(timestamps[1].template item<float>(), 101);
  EXPECT_EQ(timestamps[2].template item<float>(), 303);
}

TYPED_TEST(TGStoreTest, HandlesEmptyInputs) {
  const auto store =
      this->make_store(TestData{.src = torch::empty({0}, torch::kLong),
                                .dst = torch::empty({0}, torch::kLong),
                                .time = torch::empty({0}, torch::kLong),
                                .msg = torch::empty({0, 4}),
                                .neg_dst = std::nullopt});
  EXPECT_EQ(store->edge_count(), 0);
  EXPECT_EQ(store->node_count(), 0);
}

TYPED_TEST(TGStoreTest, SplitsWithCustomBoundaries) {
  const std::int64_t n = 10;
  const auto store =
      this->make_store(TestData{.src = torch::zeros({n}, torch::kLong),
                                .dst = torch::zeros({n}, torch::kLong),
                                .time = torch::zeros({n}, torch::kLong),
                                .msg = torch::zeros({n, 1}),
                                .val_start = 6,
                                .test_start = 8});

  EXPECT_EQ(store->train_split().start(), 0);
  EXPECT_EQ(store->train_split().end(), 6);

  EXPECT_EQ(store->val_split().start(), 6);
  EXPECT_EQ(store->val_split().end(), 8);

  EXPECT_EQ(store->test_split().start(), 8);
  EXPECT_EQ(store->test_split().end(), 10);
}

TYPED_TEST(TGStoreTest, SplitsWithNoBoundaries) {
  const std::int64_t n = 10;
  const auto store =
      this->make_store(TestData{.src = torch::zeros({n}, torch::kLong),
                                .dst = torch::zeros({n}, torch::kLong),
                                .time = torch::zeros({n}, torch::kLong),
                                .msg = torch::zeros({n, 1})});

  // Default behavior should put everything in train
  EXPECT_EQ(store->train_split().size(), 10);
  EXPECT_EQ(store->val_split().size(), 0);
  EXPECT_EQ(store->test_split().size(), 0);
}

TYPED_TEST(TGStoreTest, LabelSplitEmpty) {
  const auto store =
      this->make_store(TestData{.src = torch::zeros({5}, torch::kLong),
                                .dst = torch::zeros({5}, torch::kLong),
                                .time = torch::zeros({5}, torch::kLong),
                                .msg = torch::zeros({5, 1})});
  EXPECT_EQ(store->train_label_split().size(), 0);
  EXPECT_EQ(store->val_label_split().size(), 0);
  EXPECT_EQ(store->test_label_split().size(), 0);
}

TYPED_TEST(TGStoreTest, LabelSplitThreeDistinct) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::zeros({3}, torch::kLong),
               .label_time = torch::tensor({15, 25, 45}, torch::kLong),
               .label_target = torch::zeros({3, 1})});
  EXPECT_EQ(store->train_label_split().size(), 3);
  EXPECT_EQ(store->val_label_split().size(), 0);
  EXPECT_EQ(store->test_label_split().size(), 0);
  EXPECT_EQ(store->label_dim(), 1);
}

TYPED_TEST(TGStoreTest, LabelSplitThreeGrouped) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::zeros({3}, torch::kLong),
               .label_time = torch::tensor({15, 15, 25}, torch::kLong),
               .label_target = torch::zeros({3, 1})});
  EXPECT_EQ(store->train_label_split().size(), 2);  // 2 unique timestamps
  EXPECT_EQ(store->val_label_split().size(), 0);
  EXPECT_EQ(store->test_label_split().size(), 0);
  EXPECT_EQ(store->label_dim(), 1);
}

TYPED_TEST(TGStoreTest, LabelSplitSingle) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::zeros({1}, torch::kLong),
               .label_time = torch::tensor({15}, torch::kLong),
               .label_target = torch::zeros({1, 1})});
  EXPECT_EQ(store->train_label_split().size(), 1);
  EXPECT_EQ(store->val_label_split().size(), 0);
  EXPECT_EQ(store->test_label_split().size(), 0);
  EXPECT_EQ(store->label_dim(), 1);
}

TYPED_TEST(TGStoreTest, LabelSplitsWithCustomBoundaries) {
  // Custom split: Train [0,6), Val [6,8), Test [8,10)
  // Train ends at edge index 6 (t=70)
  // Val ends at edge index 8 (t=90)
  const auto store = this->make_store(
      TestData{.src = torch::zeros({10}, torch::kLong),
               .dst = torch::zeros({10}, torch::kLong),
               .time = torch::tensor({10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
                                     torch::kLong),
               .msg = torch::zeros({10, 1}),
               // Labels:
               // t=15 (Index 0) -> Train (15 < 70)
               // t=75 (Index 1) -> Val   (70 <= 75 < 90)
               // t=95 (Index 2) -> Test  (95 >= 90)
               .label_n_id = torch::tensor({1, 2, 3}, torch::kLong),
               .label_time = torch::tensor({15, 75, 95}, torch::kLong),
               .label_target = torch::zeros({3, 1}),
               .val_start = 6,
               .test_start = 8});

  EXPECT_EQ(store->train_label_split().start(), 0);
  EXPECT_EQ(store->train_label_split().end(), 1);

  EXPECT_EQ(store->val_label_split().start(), 1);
  EXPECT_EQ(store->val_label_split().end(), 2);

  EXPECT_EQ(store->test_label_split().start(), 2);
  EXPECT_EQ(store->test_label_split().end(), 3);
  EXPECT_EQ(store->label_dim(), 1);
}

TYPED_TEST(TGStoreTest, GetEdgeCutoffEmptyThrows) {
  const auto store =
      this->make_store(TestData{.src = torch::zeros({5}, torch::kLong),
                                .dst = torch::zeros({5}, torch::kLong),
                                .time = torch::arange(5, torch::kLong),
                                .msg = torch::zeros({5, 1})});
  EXPECT_THROW(store->get_edge_cutoff_for_label_event(0), std::out_of_range);
}

TYPED_TEST(TGStoreTest, GetEdgeCutoffThreeDistinct) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::zeros({3}, torch::kLong),
               .label_time = torch::tensor({15, 25, 45}, torch::kLong),
               .label_target = torch::zeros({3, 1})});
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(0), 1);  // Before t=20
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(1), 2);  // Before t=30
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(2), 4);  // Before t=50
}

TYPED_TEST(TGStoreTest, GetEdgeCutoffThreeGrouped) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::tensor({10, 20, 30, 40, 50}, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::zeros({3}, torch::kLong),
               .label_time = torch::tensor({15, 15, 25}, torch::kLong),
               .label_target = torch::zeros({3, 1})});

  // Stop ID for the first group (t=15)
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(0), 1);

  // Stop ID for the second group (t=25)
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(1), 2);
}

TYPED_TEST(TGStoreTest, GetEdgeCutoffSingle) {
  const auto store =
      this->make_store(TestData{.src = torch::zeros({2}, torch::kLong),
                                .dst = torch::zeros({2}, torch::kLong),
                                .time = torch::tensor({10, 20}, torch::kLong),
                                .msg = torch::zeros({2, 1}),
                                .label_n_id = torch::zeros({1}, torch::kLong),
                                .label_time = torch::tensor({25}, torch::kLong),
                                .label_target = torch::zeros({1, 1})});
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(0),
            2);  // Matches end of edges
}
TYPED_TEST(TGStoreTest, GetLabelEventEmptyThrows) {
  const auto store =
      this->make_store(TestData{.src = torch::zeros({5}, torch::kLong),
                                .dst = torch::zeros({5}, torch::kLong),
                                .time = torch::arange(5, torch::kLong),
                                .msg = torch::zeros({5, 1})});
  EXPECT_THROW(store->get_label_event(0), std::out_of_range);
}

TYPED_TEST(TGStoreTest, GetLabelEventThreeDistinct) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::arange(5, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::tensor({100, 200, 300}, torch::kLong),
               .label_time = torch::tensor({1, 2, 3}, torch::kLong),
               .label_target = torch::zeros({3, 1})});
  EXPECT_EQ(store->get_label_event(0).n_id[0].template item<std::int64_t>(),
            100);
  EXPECT_EQ(store->get_label_event(1).n_id[0].template item<std::int64_t>(),
            200);
  EXPECT_EQ(store->get_label_event(2).n_id[0].template item<std::int64_t>(),
            300);

  EXPECT_EQ(store->get_label_event(0).target[0].template item<float>(), 0);
  EXPECT_EQ(store->get_label_event(1).target[0].template item<float>(), 0);
  EXPECT_EQ(store->get_label_event(2).target[0].template item<float>(), 0);
}

TYPED_TEST(TGStoreTest, GetLabelEventThreeGrouped) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::arange(5, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::tensor({1, 2, 3}, torch::kLong),
               .label_time = torch::tensor({10, 10, 20}, torch::kLong),
               .label_target = torch::zeros({3, 1})});
  auto event = store->get_label_event(0);
  EXPECT_EQ(event.n_id.size(0), 2);  // Groups node 1 and 2
  EXPECT_EQ(event.n_id[0].template item<std::int64_t>(), 1);
  EXPECT_EQ(event.n_id[1].template item<std::int64_t>(), 2);

  event = store->get_label_event(1);
  EXPECT_EQ(event.n_id.size(0), 1);  // Groups node 1 and 2
  EXPECT_EQ(event.n_id[0].template item<std::int64_t>(), 3);
}

TYPED_TEST(TGStoreTest, GetLabelEventSingle) {
  const auto store = this->make_store(
      TestData{.src = torch::zeros({5}, torch::kLong),
               .dst = torch::zeros({5}, torch::kLong),
               .time = torch::arange(5, torch::kLong),
               .msg = torch::zeros({5, 1}),
               .label_n_id = torch::tensor({999}, torch::kLong),
               .label_time = torch::tensor({10}, torch::kLong),
               .label_target = torch::zeros({1, 1})});
  EXPECT_EQ(store->get_label_event(0).n_id[0].template item<std::int64_t>(),
            999);
  EXPECT_EQ(store->get_label_event(0).target[0].template item<float>(), 0);
}

TYPED_TEST(TGStoreTest, ValidateRejectsInvalidShapes) {
  const std::int64_t n = 10;
  const auto src = torch::zeros({n}, torch::kLong);
  const auto dst = torch::zeros({5}, torch::kLong);  // Mismatching 'n'
  const auto t = torch::zeros({n});
  const auto msg = torch::zeros({n, 4});
  const auto neg_dst = torch::zeros({n, 1}, torch::kLong);

  const auto edges = tgn::Batch{
      .src = src, .dst = dst, .time = t, .msg = msg, .neg_dst = neg_dst};
  EXPECT_THROW(std::ignore = tgn::TGStore::from_memory(edges), c10::Error);
}

TYPED_TEST(TGStoreTest, ValidateRejectsInvalidNegativesShapes) {
  const std::int64_t n = 10;
  const auto src = torch::zeros({n}, torch::kLong);
  const auto dst = torch::zeros({n}, torch::kLong);
  const auto t = torch::zeros({n}, torch::kLong);
  const auto msg = torch::zeros({n, 4});
  const auto neg_dst = torch::zeros({n}, torch::kLong);  // Should be [n, m]

  const auto edges = tgn::Batch{
      .src = src, .dst = dst, .time = t, .msg = msg, .neg_dst = neg_dst};
  EXPECT_THROW(std::ignore = tgn::TGStore::from_memory(edges), c10::Error);
}

TYPED_TEST(TGStoreTest, ValidateRejectsOutOfRangeNegatives) {
  const auto src = torch::zeros({0, 1}, torch::kLong);
  const auto dst = torch::zeros({1, 2}, torch::kLong);
  const auto t = torch::zeros({2}, torch::kLong);
  const auto msg = torch::zeros({2, 4});
  const auto neg_dst =
      torch::zeros({99}, torch::kLong);  // 99 is out of range [0, 2]

  const auto edges = tgn::Batch{
      .src = src, .dst = dst, .time = t, .msg = msg, .neg_dst = neg_dst};
  EXPECT_THROW(std::ignore = tgn::TGStore::from_memory(edges), c10::Error);
}

TYPED_TEST(TGStoreTest, ValidateRejectsFloatingPointIDs) {
  const std::int64_t n = 10;
  const auto src = torch::zeros({n});  // Float instead of long
  const auto dst = torch::zeros({n}, torch::kLong);
  const auto t = torch::zeros({n}, torch::kLong);
  const auto msg = torch::zeros({n, 4});

  const auto edges = tgn::Batch{.src = src, .dst = dst, .time = t, .msg = msg};
  EXPECT_THROW(std::ignore = tgn::TGStore::from_memory(edges), c10::Error);
}

TYPED_TEST(TGStoreTest, ValidateRejectsFloatingPointTimestamps) {
  const std::int64_t n = 10;
  const auto src = torch::zeros({n}, torch::kLong);
  const auto dst = torch::zeros({n}, torch::kLong);
  const auto t = torch::zeros({n});  // Float instead of long
  const auto msg = torch::zeros({n, 4});

  const auto edges = tgn::Batch{.src = src, .dst = dst, .time = t, .msg = msg};
  EXPECT_THROW(std::ignore = tgn::TGStore::from_memory(edges), c10::Error);
}

TEST_F(TgufTGStoreFixture, SplitResolutionUsesHeaderWhenOptsEmpty) {
  const std::int64_t n = 10;
  TestData data{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .time = torch::arange(n, torch::kLong),
      .msg = torch::zeros({n, 1}),
      .val_start = 2,  // Header value
      .test_start = 4  // Header value
  };

  // Load without splits in opts
  const auto store = make_store_with_opts(std::move(data));

  // Verify it fell back to header values
  EXPECT_EQ(store->val_split().start(), 2);
  EXPECT_EQ(store->test_split().start(), 4);
}

TEST_F(TgufTGStoreFixture, SplitResolutionOverrideHeader) {
  const std::int64_t n = 10;
  TestData data{
      .src = torch::zeros({n}, torch::kLong),
      .dst = torch::zeros({n}, torch::kLong),
      .time = torch::arange(n, torch::kLong),
      .msg = torch::zeros({n, 1}),
      .val_start = 2,  // Header value
      .test_start = 4  // Header value
  };

  // User explicitly provides different splits in opts
  const auto val_start = 6;
  const auto test_start = 9;
  const auto store =
      make_store_with_opts(std::move(data), val_start, test_start);

  // Verify Options took precedence over Header
  EXPECT_EQ(store->val_split().start(), val_start);
  EXPECT_EQ(store->test_split().start(), test_start);
}

TEST(TGRange, ValidRange) {
  const auto split = tgn::TGStore::IndexRange(1, 5);
  EXPECT_EQ(split.start(), 1);
  EXPECT_EQ(split.end(), 5);
  EXPECT_EQ(split.size(), 4);
}

TEST(TGRange, ValidEmptyRange) {
  const auto split = tgn::TGStore::IndexRange();
  EXPECT_EQ(split.start(), 0);
  EXPECT_EQ(split.end(), 0);
  EXPECT_EQ(split.size(), 0);
}

TEST(TGRange, RejectsInvalidRange) {
  EXPECT_THROW(tgn::TGStore::IndexRange(5, 3), std::out_of_range);
}
