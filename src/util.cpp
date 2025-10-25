#include "util.h"

#include <malloc.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

namespace ssg
{

  void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N)
  {
    for (unsigned i = 0; i < size; ++i)
    {
      addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i)
    {
      if (addr[i] <= addr[i - 1])
      {
        addr[i] = addr[i - 1] + 1;
      }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i)
    {
      addr[i] = (addr[i] + off) % N;
    }
  }

  float *load_data(const char *filename, unsigned &num, unsigned &dim)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
      std::cerr << "Open file error" << std::endl;
      exit(-1);
    }

    in.read((char *)&dim, 4);

    std::cout << "Dim: " << dim << '\n';
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    // num = 10000;
    std::cout << "Num: " << num << '\n';
    float *data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
      in.seekg(4, std::ios::cur);
      in.read((char *)(data + i * dim), dim * sizeof(float));
    }
    in.close();

    return data;
  }

  float *load_data(const char *filename, unsigned &num, unsigned &dim, unsigned pointed_num)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
      std::cerr << "Open file error" << std::endl;
      exit(-1);
    }

    in.read((char *)&dim, 4);

    std::cout << "Dim: " << dim << '\n';
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    num = pointed_num;
    std::cout << "Pointed Num: " << num << '\n';
    float *data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
      in.seekg(4, std::ios::cur);
      in.read((char *)(data + i * dim), dim * sizeof(float));
    }
    in.close();

    return data;
  }

  void save_data(const char *filename, unsigned num, unsigned dim, float *data_load)
  {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < num; i++)
    {
      out.write((char *)&dim, 4);
      out.write((char *)(data_load + i * dim), dim * sizeof(float));
    }
    out.close();
  }

  void save_gt(const char *filename, std::vector<std::vector<unsigned>> eps)
  {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    unsigned num = eps.size();
    for (unsigned i = 0; i < num; i++)
    {
      unsigned cnt = eps[i].size();
      out.write((char *)&cnt, sizeof(unsigned));

      out.write((char *)eps[i].data(), cnt * sizeof(unsigned));
    }
    out.close();
  }

  void load_gt(const char *filename, std::vector<std::vector<unsigned>> &eps, unsigned num)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
      std::cerr << "Open eps file error" << std::endl;
      exit(-1);
    }
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
      unsigned cnt = 0;
      in.read((char *)&cnt, sizeof(unsigned));
      eps[i].resize(cnt);
      in.read((char *)eps[i].data(), cnt * sizeof(unsigned));
    }
    in.close();
  }

  void save_id(const char *filename, std::vector<unsigned> &ids)
  {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    unsigned num = ids.size();
    for (unsigned i = 0; i < num; i++)
    {
      out.write((char *)&ids[i], sizeof(unsigned));
    }
    out.close();
  }

  void load_id(const char *filename, std::vector<unsigned> &ids, unsigned num)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
      std::cerr << "Open ids file error" << std::endl;
      exit(-1);
    }
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
      in.read((char *)&ids[i], sizeof(unsigned));
    }
    in.close();
  }

  float *data_align(float *data_ori, unsigned point_num, unsigned &dim)
  {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif
    float *data_new = 0;
    unsigned new_dim =
        (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
#ifdef __APPLE__
    data_new = new float[(size_t)new_dim * (size_t)point_num];
#else
    data_new =
        (float *)memalign(DATA_ALIGN_FACTOR * 4,
                          (size_t)point_num * (size_t)new_dim * sizeof(float));
#endif

    for (size_t i = 0; i < point_num; i++)
    {
      memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
      memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
    }

    dim = new_dim;

#ifdef __APPLE__
    delete[] data_ori;
#else
    free(data_ori);
#endif

    return data_new;
  }

  void CheckForPercent(std::vector<std::pair<double, unsigned>> &count_cal_pair, unsigned D_num)
  {
    // normalization
    double tot = 0;
    for (unsigned i = 0; i < D_num; i++)
    {
      tot += count_cal_pair[i].first;
    }
    for (unsigned i = 0; i < D_num; i++)
    {
      count_cal_pair[i].first /= tot;
    }
    double percent = 0.0;
    double limit = 0.1;
    int idx = 0;
    for (unsigned i = 0; i < D_num; i++)
    {
      percent += count_cal_pair[i].first;
      if (percent >= limit)
      {
        idx++;
        std::cout << "Filter " << i << ": " << percent << std::endl;
        limit += 0.1;
      }
    }
  }

} // namespace ssg
