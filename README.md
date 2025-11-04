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
### 0. Download the dataset
```bash
mkdir -p ./build/data && cd ./build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```



### 1. Compile
Prerequisite : openmp, opencv, cmake, boost
```bash
mkdir -p build && cd build
cmake .. && make -j
```
