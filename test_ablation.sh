cd build
cmake ..
make
cd tests

# ./test_ablation sift 200 200 12 10 100 100 50 60 50 10 1.2
./test_ablation gist 400 400 12 15 100 500 70 60 10 10 1.2
./test_ablation glove 400 420 12 15 200 500 50 60 2 10 1.2
./test_ablation crawl 400 420 12 15 100 500 40 60 2 10 1.2

./test_ablation gist 400 400 12 15 100 500 70 60 10 10 1.2
./test_ablation glove 400 420 12 15 200 500 50 60 2 10 1.2
./test_ablation crawl 400 420 12 15 100 500 40 60 2 10 1.2