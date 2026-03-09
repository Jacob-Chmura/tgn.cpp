#include <gtest/gtest.h>
#include <torch/types.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include "tgn.h"
#include "tguf.h"

class TGUFBuilderTest : public ::testing::Test {
 protected:
  std::string tguf_path_ = std::filesystem::temp_directory_path() /
                           (std::to_string(std::random_device{}()) + ".tguf");

  void TearDown() override {
    if (std::filesystem::exists(tguf_path_)) {
      std::filesystem::remove(tguf_path_);
    }
  }

  auto read_header() -> tgn::TGUFHeader {
    tgn::TGUFHeader h;
    std::ifstream is(tguf_path_, std::ios::binary);
    is.read(reinterpret_cast<char*>(&h), sizeof(tgn::TGUFHeader));
    return h;
  }
};

TEST_F(TGUFBuilderTest, PhysicalLayoutAndAlignment) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 100,
      .label_capacity = 50,
      .msg_dim = 8,
      .label_dim = 2,
      .negatives_capacity = 10,
      .negatives_per_edge = 1,
      .val_start = 70,
      .test_start = 85,
  };
  tgn::TGUFBuilder builder(schema);
  builder.finalize();

  auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.msg_dim, schema.msg_dim);
  EXPECT_EQ(h.negatives_per_edge, schema.negatives_per_edge);
  EXPECT_EQ(h.val_start, schema.val_start);
  EXPECT_EQ(h.test_start, schema.test_start);

  auto is_aligned = [](uint64_t off) { return off % tgn::TGUF_PAGE == 0; };

  EXPECT_TRUE(is_aligned(h.src_offset));
  EXPECT_TRUE(is_aligned(h.msg_offset));
  EXPECT_TRUE(is_aligned(h.label_n_id_offset));
  EXPECT_GE(h.dst_offset,
            h.src_offset + (schema.edge_capacity * sizeof(std::int64_t)));
  EXPECT_GE(h.time_offset,
            h.dst_offset + (schema.edge_capacity * sizeof(std::int64_t)));
  EXPECT_GE(h.msg_offset,
            h.time_offset + (schema.edge_capacity * sizeof(std::int64_t)));

  std::uint64_t expected_min_size =
      h.label_target_offset +
      (schema.label_capacity * schema.label_dim * sizeof(float));
  EXPECT_GE(std::filesystem::file_size(tguf_path_), expected_min_size);
}

TEST_F(TGUFBuilderTest, AppendEdges) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 2,
      .label_capacity = 0,
      .msg_dim = 2,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);

  auto batch = tgn::Batch{
      .src = torch::tensor({1, 2}, torch::kLong),
      .dst = torch::tensor({3, 4}, torch::kLong),
      .time = torch::tensor({10, 11}, torch::kLong),
      .msg = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat),
      .neg_dst = std::nullopt};

  builder.append_edges(batch);
  builder.finalize();

  const auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.version, tgn::TGUF_VERSION);
  EXPECT_EQ(h.num_edges, schema.edge_capacity);
  EXPECT_EQ(h.num_labels, schema.label_capacity);
  EXPECT_EQ(h.msg_dim, schema.msg_dim);
  EXPECT_EQ(h.label_dim, schema.label_dim);
  EXPECT_EQ(h.negatives_per_edge, schema.negatives_per_edge);
  EXPECT_EQ(h.val_start, 0);
  EXPECT_EQ(h.test_start, 0);
}

TEST_F(TGUFBuilderTest, AppendEdgesNegatives) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 4,
      .label_capacity = 0,
      .msg_dim = 2,
      .label_dim = 0,
      .negatives_capacity = 1,
      .negatives_per_edge = 1,
  };
  tgn::TGUFBuilder builder(schema);

  // First batch negatives should be skipped (since negatives_start_e_id = 1)
  auto batch = tgn::Batch{
      .src = torch::tensor({1, 2}, torch::kLong),
      .dst = torch::tensor({3, 4}, torch::kLong),
      .time = torch::tensor({10, 11}, torch::kLong),
      .msg = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat),
      .neg_dst = torch::tensor({{3}, {2}}, torch::kLong)};
  builder.append_edges(batch);

  // Second batch should be written
  batch = tgn::Batch{
      .src = torch::tensor({1, 2}, torch::kLong),
      .dst = torch::tensor({3, 4}, torch::kLong),
      .time = torch::tensor({10, 11}, torch::kLong),
      .msg = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat),
      .neg_dst = torch::tensor({{1}, {0}}, torch::kLong)};
  builder.append_edges(batch);
  builder.finalize();

  const auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.version, tgn::TGUF_VERSION);
  EXPECT_EQ(h.num_edges, schema.edge_capacity);
  EXPECT_EQ(h.num_labels, schema.label_capacity);
  EXPECT_EQ(h.msg_dim, schema.msg_dim);
  EXPECT_EQ(h.label_dim, schema.label_dim);
  EXPECT_EQ(h.num_negatives, schema.negatives_capacity);
  EXPECT_EQ(h.negatives_per_edge, schema.negatives_per_edge);
  EXPECT_EQ(h.val_start, 0);
  EXPECT_EQ(h.test_start, 0);
}

TEST_F(TGUFBuilderTest, AppendLabels) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 0,
      .label_capacity = 2,
      .msg_dim = 0,
      .label_dim = 3,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);

  auto n_id = torch::tensor({100, 200}, torch::kLong);
  auto t = torch::tensor({50, 60}, torch::kLong);
  auto y =
      torch::tensor({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}}, torch::kFloat);

  builder.append_labels(n_id, t, y);
  builder.finalize();

  const auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.version, tgn::TGUF_VERSION);
  EXPECT_EQ(h.num_edges, schema.edge_capacity);
  EXPECT_EQ(h.num_labels, schema.label_capacity);
  EXPECT_EQ(h.msg_dim, schema.msg_dim);
  EXPECT_EQ(h.label_dim, schema.label_dim);
  EXPECT_EQ(h.negatives_per_edge, schema.negatives_per_edge);
  EXPECT_EQ(h.val_start, 0);
  EXPECT_EQ(h.test_start, 0);
}

TEST_F(TGUFBuilderTest, FailureAppendEdgesAfterFinalize) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 1,
      .label_capacity = 0,
      .msg_dim = 1,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);
  builder.finalize();

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .time = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 1}),
                          .neg_dst = std::nullopt};
  EXPECT_THROW(builder.append_edges(batch), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureAppendLabelsAfterFinalize) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 0,
      .label_capacity = 1,
      .msg_dim = 1,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);
  builder.finalize();

  auto t = torch::zeros({1}, torch::kLong);
  auto y = torch::zeros({1, 1});
  EXPECT_THROW(builder.append_labels(t, t, y), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureEdgeExceedCapacity) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 2,
      .label_capacity = 0,
      .msg_dim = 1,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);

  auto batch = tgn::Batch{.src = torch::zeros({3}, torch::kLong),  // Size 3
                          .dst = torch::zeros({3}, torch::kLong),
                          .time = torch::zeros({3}, torch::kLong),
                          .msg = torch::zeros({3, 1}),
                          .neg_dst = std::nullopt};
  EXPECT_THROW(builder.append_edges(batch), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureLabelExceedCapacity) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 0,
      .label_capacity = 2,
      .msg_dim = 0,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);

  auto n_id = torch::zeros({3}, torch::kLong);  // Size 3
  auto t = torch::zeros({3}, torch::kLong);
  auto y = torch::zeros({3, 1});

  EXPECT_THROW(builder.append_labels(n_id, t, y), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureEdgeMsgDimMismatch) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 10,
      .label_capacity = 0,
      .msg_dim = 128,  // Expected 128
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .time = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 64}),  // Got 64
                          .neg_dst = std::nullopt};

  EXPECT_THROW(builder.append_edges(batch), std::invalid_argument);
}

TEST_F(TGUFBuilderTest, FailureLabelDimMismatch) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 0,
      .label_capacity = 10,
      .msg_dim = 0,
      .label_dim = 8,  // Expected 8
      .negatives_capacity = 0,
      .negatives_per_edge = 0,
  };
  tgn::TGUFBuilder builder(schema);

  auto n_id = torch::zeros({1}, torch::kLong);
  auto t = torch::zeros({1}, torch::kLong);
  auto y = torch::zeros({1, 4});  // Expected 8, got 4

  EXPECT_THROW(builder.append_labels(n_id, t, y), std::invalid_argument);
}

TEST_F(TGUFBuilderTest, FailureNegDstMissing) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 10,
      .label_capacity = 0,
      .msg_dim = 4,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 5,
  };
  tgn::TGUFBuilder builder(schema);

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .time = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 4}),
                          .neg_dst = std::nullopt};  // Expected neg dst

  EXPECT_THROW(builder.append_edges(batch), std::invalid_argument);
}

TEST_F(TGUFBuilderTest, FailureNegDstDimMisMatch) {
  const tgn::TGUFSchema schema{
      .path = tguf_path_,
      .edge_capacity = 10,
      .label_capacity = 0,
      .msg_dim = 4,
      .label_dim = 0,
      .negatives_capacity = 0,
      .negatives_per_edge = 5,
  };
  tgn::TGUFBuilder builder(schema);

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .time = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 4}),
                          .neg_dst = torch::zeros({1, 3})};  // Expected 5, got

  EXPECT_THROW(builder.append_edges(batch), std::invalid_argument);
}
