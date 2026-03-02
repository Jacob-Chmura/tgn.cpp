#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>

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
  const std::size_t num_edges = 100;
  const std::size_t msg_dim = 8;
  const std::size_t num_labels = 50;
  const std::size_t label_dim = 2;
  const std::size_t n_neg = 1;

  {
    tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                             label_dim, n_neg);
    builder.finalize();
  }

  auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.msg_dim, msg_dim);
  EXPECT_EQ(h.n_neg, n_neg);

  auto is_aligned = [](uint64_t off) { return off % tgn::TGUF_PAGE == 0; };

  EXPECT_TRUE(is_aligned(h.src_offset));
  EXPECT_TRUE(is_aligned(h.msg_offset));
  EXPECT_TRUE(is_aligned(h.label_n_id_offset));
  EXPECT_GE(h.dst_offset, h.src_offset + (num_edges * sizeof(std::int64_t)));
  EXPECT_GE(h.t_offset, h.dst_offset + (num_edges * sizeof(std::int64_t)));
  EXPECT_GE(h.msg_offset, h.t_offset + (num_edges * sizeof(std::int64_t)));

  std::uint64_t expected_min_size =
      h.label_y_true_offset + (num_labels * label_dim * sizeof(float));
  EXPECT_GE(std::filesystem::file_size(tguf_path_), expected_min_size);
}

TEST_F(TGUFBuilderTest, AppendEdges) {
  const std::size_t num_edges = 2;
  const std::size_t num_labels = 0;
  const std::size_t msg_dim = 2;
  const std::size_t label_dim = 0;
  const std::size_t n_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, n_neg);
  auto batch = tgn::Batch{
      .src = torch::tensor({1, 2}, torch::kLong),
      .dst = torch::tensor({3, 4}, torch::kLong),
      .t = torch::tensor({10, 11}, torch::kLong),
      .msg = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kFloat)};

  builder.append_edges(batch);
  builder.finalize();

  const auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.version, tgn::TGUF_VERSION);
  EXPECT_EQ(h.num_edges, num_edges);
  EXPECT_EQ(h.num_labels, num_labels);
  EXPECT_EQ(h.msg_dim, msg_dim);
  EXPECT_EQ(h.label_dim, label_dim);
  EXPECT_EQ(h.n_neg, n_neg);
}

TEST_F(TGUFBuilderTest, AppendLabels) {
  const std::size_t num_edges = 0;
  const std::size_t num_labels = 2;
  const std::size_t msg_dim = 0;
  const std::size_t label_dim = 3;
  const std::size_t n_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, n_neg);

  auto n_id = torch::tensor({100, 200}, torch::kLong);
  auto t = torch::tensor({50, 60}, torch::kLong);
  auto y =
      torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, torch::kFloat);

  builder.append_labels(n_id, t, y);
  builder.finalize();

  const auto h = read_header();
  EXPECT_EQ(h.magic, tgn::TGUF_MAGIC);
  EXPECT_EQ(h.version, tgn::TGUF_VERSION);
  EXPECT_EQ(h.num_edges, num_edges);
  EXPECT_EQ(h.num_labels, num_labels);
  EXPECT_EQ(h.msg_dim, msg_dim);
  EXPECT_EQ(h.label_dim, label_dim);
  EXPECT_EQ(h.n_neg, n_neg);
}

TEST_F(TGUFBuilderTest, FailureAppendEdgesAfterFinalize) {
  const std::size_t num_edges = 1;
  const std::size_t num_labels = 0;
  const std::size_t msg_dim = 1;
  const std::size_t label_dim = 0;
  const std::size_t n_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, n_neg);
  builder.finalize();

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .t = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 1})};
  EXPECT_THROW(builder.append_edges(batch), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureAppendLabelsAfterFinalize) {
  const std::size_t num_edges = 0;
  const std::size_t num_labels = 1;
  const std::size_t msg_dim = 0;
  const std::size_t label_dim = 1;
  const std::size_t n_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, n_neg);
  builder.finalize();

  auto t = torch::zeros({1}, torch::kLong);
  auto y = torch::zeros({1, 1});
  EXPECT_THROW(builder.append_labels(t, t, y), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureEdgeOverflow) {
  const std::size_t num_edges = 2;
  const std::size_t num_labels = 0;
  const std::size_t msg_dim = 1;
  const std::size_t label_dim = 0;
  const std::size_t num_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, num_neg);

  auto batch = tgn::Batch{.src = torch::zeros({3}, torch::kLong),  // Size 3
                          .dst = torch::zeros({3}, torch::kLong),
                          .t = torch::zeros({3}, torch::kLong),
                          .msg = torch::zeros({3, 1})};
  EXPECT_THROW(builder.append_edges(batch), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureLabelOverflow) {
  const std::size_t num_edges = 0;
  const std::size_t num_labels = 2;
  const std::size_t msg_dim = 0;
  const std::size_t label_dim = 1;
  const std::size_t num_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, num_neg);

  auto n_id = torch::zeros({3}, torch::kLong);  // Size 3
  auto t = torch::zeros({3}, torch::kLong);
  auto y = torch::zeros({3, 1});

  EXPECT_THROW(builder.append_labels(n_id, t, y), std::runtime_error);
}

TEST_F(TGUFBuilderTest, FailureEdgeMsgDimMismatch) {
  tgn::TGUFBuilder builder(tguf_path_, 10, 0, 128, 0, 0);  // Expected 128

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .t = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 64})};  // Got 64

  EXPECT_THROW(builder.append_edges(batch), std::invalid_argument);
}

TEST_F(TGUFBuilderTest, FailureLabelDimMismatch) {
  const std::size_t num_edges = 0;
  const std::size_t num_labels = 10;
  const std::size_t msg_dim = 0;
  const std::size_t label_dim = 8;
  const std::size_t num_neg = 0;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, num_neg);

  auto n_id = torch::zeros({1}, torch::kLong);
  auto t = torch::zeros({1}, torch::kLong);
  auto y = torch::zeros({1, 4});  // Expected 8, got 4

  EXPECT_THROW(builder.append_labels(n_id, t, y), std::invalid_argument);
}

TEST_F(TGUFBuilderTest, FailureNegDstMissing) {
  const std::size_t num_edges = 10;
  const std::size_t num_labels = 0;
  const std::size_t msg_dim = 4;
  const std::size_t label_dim = 0;
  const std::size_t num_neg = 5;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, num_neg);

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .t = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 4}),
                          .neg_dst = std::nullopt};  // Expected neg dst

  EXPECT_THROW(builder.append_edges(batch), std::invalid_argument);
}

TEST_F(TGUFBuilderTest, FailureNegDstDimMisMatch) {
  const std::size_t num_edges = 10;
  const std::size_t num_labels = 0;
  const std::size_t msg_dim = 4;
  const std::size_t label_dim = 0;
  const std::size_t num_neg = 5;
  tgn::TGUFBuilder builder(tguf_path_, num_edges, num_labels, msg_dim,
                           label_dim, num_neg);

  auto batch = tgn::Batch{.src = torch::zeros({1}, torch::kLong),
                          .dst = torch::zeros({1}, torch::kLong),
                          .t = torch::zeros({1}, torch::kLong),
                          .msg = torch::zeros({1, 4}),
                          .neg_dst = torch::zeros({1, 3})};  // Expected 5, got

  EXPECT_THROW(builder.append_edges(batch), std::invalid_argument);
}
