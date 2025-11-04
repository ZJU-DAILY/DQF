#include "efanna2e/index_graph.h"
#include "index_random.h"
#include "index_ssg.h"
#include "util.h"
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <iomanip>
#include <map>
#include <algorithm>

int main(int argc, char **argv)
{
  srand((unsigned int)time(NULL));
  std::string data_path_str = argv[1];
  std::string knng_path_str = argv[2];
  std::string ssg_path_str = argv[3];
  const char *data_path = data_path_str.c_str();
  const char *knng_path = knng_path_str.c_str();
  const char *ssg_path = ssg_path_str.c_str();

  unsigned K_knng = std::atoi(argv[4]);
  unsigned L_knng = std::atoi(argv[5]);
  unsigned iter = std::atoi(argv[6]);
  unsigned S_knng = std::atoi(argv[7]);
  unsigned R_knng = std::atoi(argv[8]);

  unsigned L_ssg = std::atoi(argv[9]);
  unsigned R_ssg = std::atoi(argv[10]);
  float A = std::atof(argv[11]);

  unsigned n_try = 10;
  unsigned points_num, dim;
  float *data_load = nullptr;
  data_load = ssg::load_data(data_path, points_num, dim);
  data_load = ssg::data_align(data_load, points_num, dim);
  unsigned D_num = points_num * 0.9;
  unsigned Q_num = points_num - D_num;
  std::cout << "D_num: " << D_num << std::endl;
  std::cout << "Q_num: " << Q_num << std::endl;
  float *D_load = new float[D_num * dim];
  for (unsigned i = Q_num; i < points_num; ++i)
  {
    std::memcpy(D_load + (i - Q_num) * dim, data_load + i * dim, dim * sizeof(float));
  }
  free(data_load);

  ssg::IndexRandom init_index_knng(dim, D_num);
  efanna2e::IndexGraph index_knng(dim, D_num, efanna2e::FAST_L2, (efanna2e::Index *)(&init_index_knng));
  efanna2e::Parameters paras_knng;
  paras_knng.Set<unsigned>("K", K_knng);
  paras_knng.Set<unsigned>("L", L_knng);
  paras_knng.Set<unsigned>("iter", iter);
  paras_knng.Set<unsigned>("S", S_knng);
  paras_knng.Set<unsigned>("R", R_knng);
  
  ssg::IndexRandom init_index_ssg(dim, D_num);
  ssg::IndexSSG index_ssg(dim, D_num, ssg::FAST_L2,
                          (ssg::Index *)(&init_index_ssg));
  ssg::Parameters paras_ssg;
  paras_ssg.Set<unsigned>("L", L_ssg);
  paras_ssg.Set<unsigned>("R", R_ssg);
  paras_ssg.Set<float>("A", A);
  paras_ssg.Set<unsigned>("n_try", n_try);
  paras_ssg.Set<std::string>("nn_graph_path", knng_path_str);

  std::cout << "Build KNNG..." << std::endl;
  index_knng.Build(D_num, D_load, paras_knng);
  index_knng.Save(knng_path);
  std::cout << "Build SSG..." << std::endl;
  index_ssg.Build(D_num, D_load, paras_ssg);
  index_ssg.Save(ssg_path);
}