//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#include <index.h>
namespace ssg {
Index::Index(const size_t dimension, const size_t n, Metric metric = L2)
    : dimension_(dimension), nd_(n), has_built(false) {
  switch (metric) {
    case L2:
      distance_ = new DistanceL2();
      break;
    default:
      distance_ = new DistanceL2();
      break;
  }
}

Index::Index(const size_t dimension, const size_t n, const size_t s_nd, Metric metric = L2)
    : dimension_(dimension), nd_(n), s_nd_(s_nd), has_built(false) {
  switch (metric) {
    case L2:
      distance_ = new DistanceL2();
      break;
    default:
      distance_ = new DistanceL2();
      break;
  }
}
Index::~Index() {}
}  // namespace ssg
