#include <cstdint>
#include <iostream>

#include "tgn.h"

auto main() -> int {
  const std::string tguf_path = "data/test.tguf";
  std::cout << "Building TGUF file: " << tguf_path << "..." << std::endl;

  const std::size_t n_edges = 0;
  const std::size_t n_labels = 0;
  const std::size_t n_neg = 0;
  const std::size_t m_dim = 0;
  const std::size_t l_dim = 0;

  tgn::TGUFBuilder builder(tguf_path, n_edges, n_labels, m_dim, l_dim, n_neg);

  // tgn::Batch batch{.src = data.src,
  //                  .dst = data.dst,
  //                  .t = data.t,
  //                  .msg = data.msg,
  //                  .neg_dst = data.neg_dst};
  // builder.append_edges(batch);
  // builder.append_labels(*data.label_n_id, *data.label_t, *data.label_y_true);
  builder.finalize();
  std::cout << "TGUF construction complete." << std::endl;
}
