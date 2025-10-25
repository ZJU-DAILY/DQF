cd build
cmake ..
make
cd tests

./test_query_shift sift 200 200 12 10 100 100 50 60 50 10 19 1.2
./test_query_shift gist 400 400 12 15 100 500 70 60 10 10 50 1.2
./test_query_shift glove 400 420 12 15 200 500 50 60 2 10 110 1.2
./test_query_shift crawl 400 420 12 15 100 500 40 60 2 10 40 1.2