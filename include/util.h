#ifndef SSG_UTIL_H
#define SSG_UTIL_H

#include <random>

namespace ssg
{

    void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N);

    float *load_data(const char *filename, unsigned &num, unsigned &dim);

    float *load_data(const char *filename, unsigned &num, unsigned &dim, unsigned pointed_num);

    float *data_align(float *data_ori, unsigned point_num, unsigned &dim);

    void save_data(const char *filename, unsigned num, unsigned dim, float *data_load);

    void save_gt(const char *filename, std::vector<std::vector<unsigned>> eps);

    void load_gt(const char *filename, std::vector<std::vector<unsigned>> &eps, unsigned num);

    void save_id(const char *filename, std::vector<unsigned> &ids);

    void load_id(const char *filename, std::vector<unsigned> &ids, unsigned num);

    void CheckForPercent(std::vector<std::pair<double, unsigned>> &count_cal_pair, unsigned D_num);

} // namespace ssg

#endif // EFANNA2E_UTIL_H
