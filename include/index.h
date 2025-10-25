//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef SSG_INDEX_H
#define SSG_INDEX_H

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace ssg
{

  class Index
  {
  public:
    explicit Index(const size_t dimension, const size_t n, Metric metric);
    explicit Index(const size_t dimension, const size_t n, const size_t s_nd, Metric metric);

    virtual ~Index();

    virtual void Build(size_t n, const float *data,
                       const Parameters &parameters) = 0;

    virtual void Search(const float *query, const float *x, size_t k,
                        const Parameters &parameters, unsigned *indices) = 0;

    virtual void Save(const char *filename) = 0;

    virtual void Load(const char *filename) = 0;

    inline bool HasBuilt() const { return has_built; }

    inline size_t GetDimension() const { return dimension_; };

    inline size_t GetSizeOfDataset() const { return nd_; }

    inline const float *GetDataset() const { return data_; }

  protected:
    const size_t dimension_;
    const float *data_;
    const float *s_data_;
    size_t nd_;
    size_t s_nd_;
    bool has_built;
    Distance *distance_;
  };

} // namespace ssg

#endif // EFANNA2E_INDEX_H
