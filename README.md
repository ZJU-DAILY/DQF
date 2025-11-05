# DQF
Under construcion...

DQF (Dual-Index Query Framework) is a novel high-dimensional approximate nearest neighbor search framework that leverages a dual-layer index structure and dynamic search strategy based on a decision tree. It optimizes search efficiency by prioritizing high-frequency queries through a "hot index" and managing dynamic query distributions with an incremental update mechanism. This code forked off from [code for NSSG](https://github.com/ZJULearning/ssg) algorithm.

Benchmark datasets
------

| Data set  | Download                 | dimension | nb base vectors | nb query vectors | original website                                               |
|-----------|--------------------------|-----------|-----------------|------------------|----------------------------------------------------------------|
| SIFT1M    |[original website](http://corpus-texmex.irisa.fr/)| 128       | 1,000,000       | 10,000           | [original website](http://corpus-texmex.irisa.fr/)             |
| GIST1M    |[original website](http://corpus-texmex.irisa.fr/)| 128       | 1,000,000       | 1,000            | [original website](http://corpus-texmex.irisa.fr/)             |
| Crawl     | [crawl.tar.gz](http://downloads.zjulearning.org.cn/data/crawl.tar.gz) (1.7GB)     | 300       | 1,989,995       | 10,000           | [original website](http://commoncrawl.org/)                    |
| GloVe-100 | [glove-100.tar.gz](http://downloads.zjulearning.org.cn/data/glove-100.tar.gz) (424MB) | 100       | 1,183,514       | 10,000           | [original website](https://nlp.stanford.edu/projects/glove/)   |


How to use
------
### 0. Download the dataset

Here, we use SIFT1M as an example.

```bash
mkdir -p ./build/data && cd ./build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
```

### 1. Compile
Prerequisite : openmp, opencv, cmake, boost
```bash
cmake ..
make -j
```

### 2. Build the full index
For example:

```bash
cd /path/to/project/build/tests/
./test_build_full_index data_path knn_graph_path ssg_path K L_KNNG iter S R_KNNG L_SSG R_SSG Angle
```

* **data\_path**: Path to the original data.
* **knn\_graph\_path**: Path to the pre-built kNN graph.
* **ssg\_path**: Path where the resulting SSG is stored.
* **K, L\_KNNG, iter, S, R\_KNNG**: Parameters of [EFANNA](https://github.com/ZJULearning/efanna_graph).
* **L\_SSG, R\_SSG, Angle**: Parameters of [NSSG](https://github.com/ZJULearning/ssg).

To test all datasets, the Bash commands are as follows:
```bash
./test_build_full_index ../data/sift/sift_base.fvecs ./sift.knng ./sift.ssg 200 200 12 10 100 100 50 60
```

### 3. Search
For example:
```bash
cd /path/to/project/build/tests/
./test_search data_path ssg_path hot_knn_graph_path hot_ssg_path K L_KNNG iter S R_KNNG L_SSG R_SSG Angle Beta
```

* **data\_path**: Path to the original data.
* **ssg\_path**: Path to the pre-built full SSG graph.
* **K, L\_KNNG, iter, S, R\_KNNG**: Parameters of [EFANNA](https://github.com/ZJULearning/efanna_graph), built for the hot index (same as the full index).
* **L\_SSG, R\_SSG, Angle**: Parameters of [NSSG](https://github.com/ZJULearning/ssg), built for the hot index (same as the full index).
* **Beta**: Used to calculate the hot index S_L = K + (L - K) / beta.
* **K**: Number of final nearest neighbors.
* **Alpha**: Parameter of the Zipf distribution (default: 1.2).

```bash
./test_search ../data/sift/sift_base.fvecs sift.ssg sift_s.knng sift_s.ssg 200 200 12 10 100 100 50 60 50 10 1.2
```

### 4. Test query shift
```bash
./test_query_shift ../data/sift/sift_base.fvecs sift.ssg sift_s.knng sift_s.ssg 200 200 12 10 100 100 50 60 50 10 20 1.2
```
