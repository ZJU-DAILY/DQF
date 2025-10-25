#ifndef SSG_INDEX_SSG_H
#define SSG_INDEX_SSG_H

#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "util.h"
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
namespace ssg
{
  struct TrainData
  {
    double query_index_dist_1st;
    double query_index_dist_1st_div_kth;
    double dist_1st;
    double dist_1st_div_kth;
    unsigned dist_cnt;
    unsigned update_cnt;
  };
  class IndexSSG : public Index
  {
  public:
    explicit IndexSSG(const size_t dimension, const size_t n, Metric m,
                      Index *initializer);
    explicit IndexSSG(const size_t dimension, const size_t n, const size_t s_n, Metric m,
                      Index *initializer);

    virtual ~IndexSSG();

    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;

    virtual void Build(size_t n, const float *data,
                       const Parameters &parameters) override;

    virtual void Search(const float *query, const float *x, size_t k,
                        const Parameters &parameters, unsigned *indices) override;
    void SearchWithOptGraph(const float *query, size_t K,
                            const Parameters &parameters, unsigned *indices);
    void OptimizeGraph(const float *data);
    void OptimizeGraphWithSGraph(const float *data, const float *s_data);
    std::chrono::duration<double> diff;
    void SearchWithEps(const float *query, const float *x, size_t K,
                       const Parameters &parameters, unsigned *indices, unsigned *ep, std::vector<std::vector<unsigned>> &eps_set, unsigned K_small);
    void LoadWithSGraph(const char *filename, const char *s_filename);
    void BuildWithEps(size_t n, const float *data,
                      const Parameters &parameters, std::vector<unsigned> &eps);
    unsigned GetTrainData(const float *query, size_t K,
                          const Parameters &parameters, unsigned *indices, unsigned *ids, std::vector<TrainData> &train_data);
    void SearchWithDtree(const float *query, size_t K,
                         const Parameters &parameters, unsigned *indices, unsigned *ids, cv::Ptr<cv::ml::DTrees> dtree);
    void SearchWithDtree_AddStep(const float *query, size_t K,
                                 const Parameters &parameters, unsigned *indices, unsigned *ids, cv::Ptr<cv::ml::DTrees> dtree);
    void FreeOptimizeGraph();
    void Insert(const float *data, size_t be, size_t ed,
                const Parameters &parameters);
    void insert_prune(unsigned q, std::vector<Neighbor> &pool,
                                const Parameters &parameters, float threshold);
    void SearchForCandidate(const float *query, size_t K,
                                    const Parameters &parameters,
                                    std::vector<Neighbor> &indices, size_t be, boost::dynamic_bitset<> &flags_build);
    void OptimizeGraphWithSGraph_MoreSpace(const float *data, const float *s_data, unsigned more_space);
    void SearchWithoutDtree(const float *query, size_t K,
                                 const Parameters &parameters, unsigned *indices, unsigned *ids);

  protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    typedef std::vector<SimpleNeighbors> LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;
    CompactGraph s_final_graph_;
    Index *initializer_;

    void init_graph(const Parameters &parameters);
    void get_neighbors(const float *query, const Parameters &parameter,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset);
    void get_neighbors(const unsigned q, const Parameters &parameter,
                       std::vector<Neighbor> &pool);
    void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                    const Parameters &parameter, float threshold,
                    SimpleNeighbor *cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);
    void InterInsert(unsigned n, unsigned range, float threshold,
                     std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_);
    void Load_nn_graph(const char *filename);
    void strong_connect(const Parameters &parameter);

    void DFS(boost::dynamic_bitset<> &flag,
             std::vector<std::pair<unsigned, unsigned>> &edges, unsigned root,
             unsigned &cnt);
    bool check_edge(unsigned u, unsigned t);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                  const Parameters &parameter);
    void DFS_expand(const Parameters &parameter);
    void DFS_expand_with_eps(const Parameters &parameter, std::vector<unsigned> &eps);

  private:
    unsigned width;
    unsigned ep_; // not in use
    std::vector<unsigned> eps_;
    std::vector<std::unique_ptr<std::mutex>> locks;
    char *opt_graph_;
    char *s_opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    size_t s_node_size;
    size_t s_data_len;
    size_t s_neighbor_len;
    KNNGraph nnd_graph;
    unsigned s_width;
    std::vector<unsigned> s_eps_;
  };

} // namespace ssg

#endif
