//
// Created by 付聪 on 2017/6/21.
//

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

std::vector<float> generate_query_probabilities(int num_vectors, float alpha)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> uniform_dist(0.0, 1.0);
  std::vector<float> probabilities(num_vectors);
  for (int i = 0; i < num_vectors; ++i)
  {
    probabilities[i] = 1.0f / std::pow(i + 1, alpha);
  }
  // shuffle the probabilities
  std::shuffle(probabilities.begin(), probabilities.end(), gen);

  float sum = 0.0f;
  for (float p : probabilities)
    sum += p;
  for (float &p : probabilities)
    p /= sum;

  return probabilities;
}

std::vector<size_t> generate_query_ids(int num_queries, const std::vector<float> &probabilities)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
  std::vector<size_t> query_ids(num_queries);
  for (int i = 0; i < num_queries; ++i)
  {
    query_ids[i] = dist(gen);
  }
  return query_ids;
}

int main(int argc, char **argv)
{
  srand((unsigned int)time(NULL));
  std::string data_path_str = argv[1];
  std::string ssg_path_str = argv[2];
  std::string s_knng_path_str = argv[3];
  std::string s_ssg_path_str = argv[4];

  const char *data_path = data_path_str.c_str();
  const char *ssg_path = ssg_path_str.c_str();
  const char *s_knng_path = s_knng_path_str.c_str();
  const char *s_ssg_path = s_ssg_path_str.c_str();

  unsigned K_knng = std::atoi(argv[5]);
  unsigned L_knng = std::atoi(argv[6]);
  unsigned iter = std::atoi(argv[7]);
  unsigned S_knng = std::atoi(argv[8]);
  unsigned R_knng = std::atoi(argv[9]);

  unsigned L_ssg = std::atoi(argv[10]);
  unsigned R_ssg = std::atoi(argv[11]);
  float A = std::atof(argv[12]);
  unsigned n_try = 10;

  unsigned points_num, dim;
  float *data_load = nullptr;
  data_load = ssg::load_data(data_path, points_num, dim);
  data_load = ssg::data_align(data_load, points_num, dim);
  unsigned D_num = points_num * 0.9;
  unsigned Q_num = points_num - D_num;
  unsigned QD_num = D_num * 0.005;
  unsigned Qhis_num = Q_num * 10;
  unsigned Qtest_num = 1000;
  unsigned train_num = 10000;
  unsigned beta = std::atoi(argv[13]);
  unsigned K = std::atoi(argv[14]);
  unsigned L = K;
  double alpha = std::atof(argv[15]);
  unsigned S_L = K + (L - K) / beta;

  std::vector<float> probabilities = generate_query_probabilities(Q_num, alpha);
  std::vector<size_t> Qhis = generate_query_ids(Qhis_num, probabilities);
  std::vector<size_t> Qtest = generate_query_ids(Qtest_num, probabilities);

  float *D_load = new float[D_num * dim];
  float *Q_load = new float[Q_num * dim];

  for (unsigned i = 0; i < Q_num; ++i)
  {
    std::memcpy(Q_load + i * dim, data_load + i * dim, dim * sizeof(float));
  }
  for (unsigned i = Q_num; i < points_num; ++i)
  {
    std::memcpy(D_load + (i - Q_num) * dim, data_load + i * dim, dim * sizeof(float));
  }
  free(data_load);

  unsigned add_L = 5;
  double last_recall = 0;
  while (true)
  {

    std::cout << "L: " << L << " " << "S_L: " << S_L << std::endl;
    std::vector<unsigned> ids;
    float *QD_load = new float[QD_num * dim];
    int pos = 0;

    ssg::IndexRandom init_index_DQF(dim, D_num);
    ssg::IndexSSG index_DQF(dim, D_num, QD_num, ssg::FAST_L2,
                            (ssg::Index *)(&init_index_DQF));

    std::set<unsigned> unique_Qhis(Qhis.begin(), Qhis.end());
    std::vector<unsigned> train_ids(unique_Qhis.begin(), unique_Qhis.end());
    train_num = std::min(train_num, (unsigned)train_ids.size());
    train_ids.resize(train_num);

    std::vector<unsigned> count_cal(D_num, 0);
    ssg::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("S_L_search", S_L);
    paras.Set<unsigned>("freq", 50);
    std::vector<std::vector<unsigned>> res(Qhis_num);
    ssg::IndexRandom init_index(dim, D_num);
    ssg::IndexSSG index(dim, D_num, ssg::FAST_L2,
                        (ssg::Index *)(&init_index));
    index.Load(ssg_path);
    index.OptimizeGraph(D_load);
    for (unsigned i = 0; i < Qhis_num; i++)
      res[i].resize(K);
#pragma omp parallel for
    for (unsigned i = 0; i < Qhis_num; i++)
    {
      index.SearchWithOptGraph(Q_load + Qhis[i] * dim, K, paras, res[i].data());
    }

    for (unsigned i = 0; i < Qhis_num; i++)
    {
      for (unsigned j = 0; j < K; j++)
      {
        count_cal[res[i][j]] += 1;
      }
    }
    std::vector<std::pair<double, unsigned>> count_cal_pair(D_num);
    for (unsigned i = 0; i < D_num; i++)
    {
      count_cal_pair[i].first = (double)count_cal[i];
      count_cal_pair[i].second = i;
    }
    sort(count_cal_pair.begin(), count_cal_pair.end(), std::greater<std::pair<unsigned, unsigned>>());
    for (unsigned i = 0; i < QD_num; i++)
    {
      ids.push_back(count_cal_pair[i].second);
    }
    for (unsigned i = 0; i < QD_num; ++i)
    {
      std::memcpy(QD_load + pos * dim, D_load + count_cal_pair[i].second * dim, dim * sizeof(float));
      pos++;
    }

    ssg::Distance *dist_func = new ssg::DistanceL2();
    std::vector<std::vector<unsigned>> QA(Qtest_num, std::vector<unsigned>(K));
#pragma omp parallel for
    for (unsigned i = 0; i < Qtest_num; ++i)
    {
      float *query = Q_load + Qtest[i] * dim;
      std::vector<std::pair<float, unsigned>> distances(D_num);
      for (unsigned j = 0; j < D_num; ++j)
      {
        float *point = D_load + j * dim;
        float dist = 0;
        for (unsigned d = 0; d < dim; ++d)
        {
          dist += (query[d] - point[d]) * (query[d] - point[d]);
        }
        distances[j] = {dist, j};
      }
      std::sort(distances.begin(), distances.end());
      for (unsigned j = 0; j < K; ++j)
      {
        QA[i][j] = distances[j].second;
      }
    }

    ssg::Parameters paras_ssg;
    std::string nn_graph_path(s_knng_path);
    paras_ssg.Set<unsigned>("L", L_ssg);
    paras_ssg.Set<unsigned>("R", R_ssg);
    paras_ssg.Set<float>("A", A);
    paras_ssg.Set<unsigned>("n_try", n_try);
    paras_ssg.Set<std::string>("nn_graph_path", nn_graph_path);
    ssg::IndexRandom init_index_knng(dim, QD_num);
    efanna2e::IndexGraph index_knng(dim, QD_num, efanna2e::FAST_L2, (efanna2e::Index *)(&init_index_knng));
    efanna2e::Parameters paras_knng;
    paras_knng.Set<unsigned>("K", K_knng);
    paras_knng.Set<unsigned>("L", L_knng);
    paras_knng.Set<unsigned>("iter", iter);
    paras_knng.Set<unsigned>("S", S_knng);
    paras_knng.Set<unsigned>("R", R_knng);
    index_knng.Build(QD_num, QD_load, paras_knng);
    index_knng.Save(s_knng_path);

    ssg::IndexRandom init_index_ssg(dim, QD_num);
    ssg::IndexSSG index_ssg(dim, QD_num, ssg::FAST_L2,
                            (ssg::Index *)(&init_index_ssg));

    index_ssg.Build(QD_num, QD_load, paras_ssg);
    index_ssg.Save(s_ssg_path);

    index_DQF.LoadWithSGraph(ssg_path, s_ssg_path);
    index_DQF.OptimizeGraphWithSGraph(D_load, QD_load);

    for (unsigned i = 0; i < train_num; i++)
      res[i].resize(K);
    std::vector<ssg::TrainData> train_data;
    std::vector<ssg::TrainData> all_train_data;
    std::vector<unsigned> trainedPointsMarkers;
    for (unsigned i = 0; i < train_num; i++)
    {
      unsigned stop_pos = index_DQF.GetTrainData(Q_load + train_ids[i] * dim, K, paras, res[i].data(), ids.data(), train_data);
      for (auto &td : train_data)
      {
        all_train_data.push_back(td);
        int trainedPointsMarker = 0;
        if (stop_pos < td.dist_cnt)
          trainedPointsMarker = 1;
        trainedPointsMarkers.push_back(trainedPointsMarker);
      }
      train_data.clear();
    }

    std::vector<float> features;
    std::vector<int> labels;
    for (size_t i = 0; i < all_train_data.size(); ++i)
    {
      features.push_back(all_train_data[i].query_index_dist_1st);
      features.push_back(all_train_data[i].query_index_dist_1st_div_kth);
      features.push_back(all_train_data[i].dist_1st);
      features.push_back(all_train_data[i].dist_1st_div_kth);
      features.push_back(all_train_data[i].dist_cnt);
      features.push_back(all_train_data[i].update_cnt);
      labels.push_back(trainedPointsMarkers[i]);
    }

    cv::Mat trainData(features.size() / 6, 6, CV_32F, features.data());
    cv::Mat responses(labels.size(), 1, CV_32SC1, labels.data());
    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    dtree->setMaxDepth(10);
    dtree->setMinSampleCount(10);
    dtree->setCVFolds(0);
    dtree->setUseSurrogates(false);
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);

    cv::Ptr<cv::ml::TrainData>
        trainDataObj = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, responses);

    if (dtree->train(trainDataObj))
    {
      dtree->save("decision_tree_model.yml");
    }

    for (unsigned i = 0; i < Qtest_num; i++)
    {
      index.SearchWithOptGraph(Q_load + Qtest[i] * dim, K, paras, res[i].data());
    }

    double pre_recall = 0.0;
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < Qtest_num; i++)
    {
      index.SearchWithOptGraph(Q_load + Qtest[i] * dim, K, paras, res[i].data());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    for (unsigned i = 0; i < Qtest_num; ++i)
    {
      std::set<unsigned> result(res[i].begin(), res[i].end());
      for (unsigned j = 0; j < K; ++j)
      {
        if (result.count(QA[i][j]))
        {
          pre_recall += 1.0;
        }
      }
    }
    pre_recall = pre_recall / (Qtest_num * K);
    float pre_qps = Qtest_num / diff.count();
    std::cout << "NSSG QPS:" << Qtest_num / diff.count() << ',';
    std::cout << "NSSG Recall: " << pre_recall << '\n';


    for (unsigned i = 0; i < Qtest_num; i++)
      res[i].resize(K);
    double ab_recall = 0.0;
    s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < Qtest_num; i++)
    {
      index_DQF.SearchWithoutDtree(Q_load + Qtest[i] * dim, K, paras, res[i].data(), ids.data());
    }
    e = std::chrono::high_resolution_clock::now();
    diff = e - s;
    for (unsigned i = 0; i < Qtest_num; ++i)
    {
      std::set<unsigned> result(res[i].begin(), res[i].end());
      for (unsigned j = 0; j < K; ++j)
      {
        if (result.count(QA[i][j]))
        {
          ab_recall += 1.0;
        }
      }
    }
    ab_recall = ab_recall / (Qtest_num * K);
    float ab_qps = Qtest_num / diff.count();
    std::cout << "Ablation QPS: " << ab_qps << ',';
    std::cout << "Ablation Recall: " << ab_recall << '\n';

    for (unsigned i = 0; i < Qtest_num; i++)
      res[i].resize(K);
    double recall = 0.0;
    s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < Qtest_num; i++)
    {
      index_DQF.SearchWithDtree(Q_load + Qtest[i] * dim, K, paras, res[i].data(), ids.data(), dtree);
    }
    e = std::chrono::high_resolution_clock::now();
    diff = e - s;
    for (unsigned i = 0; i < Qtest_num; ++i)
    {
      std::set<unsigned> result(res[i].begin(), res[i].end());
      for (unsigned j = 0; j < K; ++j)
      {
        if (result.count(QA[i][j]))
        {
          recall += 1.0;
        }
      }
    }
    recall = recall / (Qtest_num * K);
    float qps = Qtest_num / diff.count();
    std::cout << "DQF QPS: " << qps << ',';
    std::cout << "DQF Recall: " << recall << '\n';
    if (pre_recall > 0.99)
      break;
    if (pre_recall - last_recall < 0.01)
    {
      add_L *= 2;
    }
    L += add_L;
    S_L = K + (L - K) / beta;
    index.FreeOptimizeGraph();
    index_DQF.FreeOptimizeGraph();
    last_recall = pre_recall;
  }
}