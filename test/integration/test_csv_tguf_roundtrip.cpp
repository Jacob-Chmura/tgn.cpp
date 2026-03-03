#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "tgn.h"

class CSV_TGUF_RoundtripTest : public ::testing::Test {
 protected:
  const std::string resource_path =
      "test/integration/resources/csv_tguf_roundtrip";
  const std::string python_tool_path = "tools/convert_csv_to_tguf.py";
  const std::string edges_csv = resource_path + "/edges.csv";
  const std::string labels_csv = resource_path + "/labels.csv";
  const std::string output_tguf = resource_path + "/out.tguf";

  void SetUp() override {
    std::string cmd = "uv run python " + python_tool_path;
    cmd += " --edges " + edges_csv;
    cmd += " --labels " + labels_csv;
    cmd += " --output " + output_tguf;
    ASSERT_EQ(std::system(cmd.c_str()), 0);
  }
};

TEST_F(CSV_TGUF_RoundtripTest, Verify) {
  auto store = tgn::TGStore::from_tguf({.path = output_tguf});

  // TODO(kuba): Would be nice to read in expect out from resources
  EXPECT_EQ(store->num_edges(), 3);
  EXPECT_EQ(store->num_nodes(), 3001);
  EXPECT_EQ(store->msg_dim(), 2);
  EXPECT_EQ(store->label_dim(), 2);

  // Check Edges
  auto batch = store->get_batch(0, 3, tgn::NegStrategy::PreComputed);

  EXPECT_EQ(batch.src[0].item<std::int64_t>(), 1);
  EXPECT_EQ(batch.src[1].item<std::int64_t>(), 2);
  EXPECT_EQ(batch.src[2].item<std::int64_t>(), 3);

  EXPECT_EQ(batch.dst[0].item<std::int64_t>(), 2000);
  EXPECT_EQ(batch.dst[1].item<std::int64_t>(), 3000);
  EXPECT_EQ(batch.dst[2].item<std::int64_t>(), 1000);

  EXPECT_EQ(batch.t[0].item<std::int64_t>(), 100.0);
  EXPECT_EQ(batch.t[1].item<std::int64_t>(), 200.0);
  EXPECT_EQ(batch.t[2].item<std::int64_t>(), 300.0);

  EXPECT_FLOAT_EQ(batch.msg[0][0].item<float>(), 0.11F);
  EXPECT_FLOAT_EQ(batch.msg[2][0].item<float>(), 0.31F);

  // Check Pre-Computed Negatives
  EXPECT_EQ(batch.neg_dst.value()[0][0].item<std::int64_t>(), 999);
  EXPECT_EQ(batch.neg_dst.value()[0][1].item<std::int64_t>(), 888);

  EXPECT_EQ(batch.neg_dst.value()[2][0].item<std::int64_t>(), 555);
  EXPECT_EQ(batch.neg_dst.value()[2][1].item<std::int64_t>(), 444);

  // Check labels
  auto label0 = store->get_label_event(0);
  EXPECT_EQ(label0.n_id[0].item<std::int64_t>(), 1);
  EXPECT_FLOAT_EQ(label0.y_true[0].item<float>(), 1.0F);
  EXPECT_FLOAT_EQ(label0.y_true[1].item<float>(), 0.0F);

  auto label1 = store->get_label_event(1);
  EXPECT_EQ(label1.n_id[0].item<std::int64_t>(), 2);
  EXPECT_FLOAT_EQ(label1.y_true[0].item<float>(), 0.0F);
  EXPECT_FLOAT_EQ(label1.y_true[1].item<float>(), 1.0F);

  // Verify Edge-Label Synchronization (Stop IDs)
  // Label 0 (t=250.0) sees edges before 250.0 (Index 0, 1) -> Stop ID = 2
  EXPECT_EQ(store->get_stop_e_id_for_label_event(0), 2);
  // Label 1 (t=350.0) sees all edges (Index 0, 1, 2) -> Stop ID = 3
  EXPECT_EQ(store->get_stop_e_id_for_label_event(1), 3);
}
