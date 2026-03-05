#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "tgn.h"

class CSV_TGUF_RoundtripTest : public ::testing::Test {
 protected:
  const std::string script_path = "scripts/convert_csv_to_tguf.sh";
  const std::string resource_dir =
      "test/integration/resources/csv_tguf_roundtrip/";
  const std::string edges_csv = resource_dir + "edges.csv";
  const std::string labels_csv = resource_dir + "labels.csv";
  const std::string output_tguf = resource_dir + "out.tguf";
  const std::string cmd = std::format("{} {} {} {}", script_path, edges_csv,
                                      output_tguf, labels_csv);

  void SetUp() override { ASSERT_EQ(std::system(cmd.c_str()), 0); }
};

TEST_F(CSV_TGUF_RoundtripTest, Verify) {
  auto store = tgn::TGStore::from_tguf(output_tguf);

  // TODO(kuba): Would be nice to read in expect out from resources
  EXPECT_EQ(store->edge_count(), 3);
  EXPECT_EQ(store->node_count(), 31);
  EXPECT_EQ(store->msg_dim(), 2);
  EXPECT_EQ(store->label_dim(), 2);

  // Check Edges
  auto batch = store->get_batch(0, 3, tgn::NegStrategy::PreComputed);

  EXPECT_EQ(batch.src[0].item<std::int64_t>(), 1);
  EXPECT_EQ(batch.src[1].item<std::int64_t>(), 2);
  EXPECT_EQ(batch.src[2].item<std::int64_t>(), 3);

  EXPECT_EQ(batch.dst[0].item<std::int64_t>(), 20);
  EXPECT_EQ(batch.dst[1].item<std::int64_t>(), 30);
  EXPECT_EQ(batch.dst[2].item<std::int64_t>(), 10);

  EXPECT_EQ(batch.time[0].item<std::int64_t>(), 5);
  EXPECT_EQ(batch.time[1].item<std::int64_t>(), 10);
  EXPECT_EQ(batch.time[2].item<std::int64_t>(), 15);

  EXPECT_FLOAT_EQ(batch.msg[0][0].item<float>(), 0.11F);
  EXPECT_FLOAT_EQ(batch.msg[2][0].item<float>(), 0.31F);

  // Check Pre-Computed Negatives
  EXPECT_EQ(batch.neg_dst.value()[0][0].item<std::int64_t>(), 9);
  EXPECT_EQ(batch.neg_dst.value()[0][1].item<std::int64_t>(), 8);
  EXPECT_EQ(batch.neg_dst.value()[1][0].item<std::int64_t>(), 7);
  EXPECT_EQ(batch.neg_dst.value()[1][1].item<std::int64_t>(), 6);
  EXPECT_EQ(batch.neg_dst.value()[2][0].item<std::int64_t>(), 5);
  EXPECT_EQ(batch.neg_dst.value()[2][1].item<std::int64_t>(), 4);

  // Check labels
  auto label0 = store->get_label_event(0);
  EXPECT_EQ(label0.n_id[0].item<std::int64_t>(), 1);
  EXPECT_FLOAT_EQ(label0.target[0][0].item<float>(), 1.0F);
  EXPECT_FLOAT_EQ(label0.target[0][1].item<float>(), 0.0F);

  auto label1 = store->get_label_event(1);
  EXPECT_EQ(label1.n_id[0].item<std::int64_t>(), 2);
  EXPECT_FLOAT_EQ(label1.target[0][0].item<float>(), 0.0F);
  EXPECT_FLOAT_EQ(label1.target[0][1].item<float>(), 1.0F);

  // Check Edge-Label Synchronization (Stop IDs)
  // Label 0 (t=12) sees edges before 12 (Index 0, 1) -> Stop ID = 2
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(0), 2);
  // Label 1 (t=24) sees all edges (Index 0, 1, 2) -> Stop ID = 3
  EXPECT_EQ(store->get_edge_cutoff_for_label_event(1), 3);
}
