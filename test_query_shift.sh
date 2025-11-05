cd build
cmake ..
make
cd tests

./test_query_shift ../data/sift/sift_base.fvecs sift.ssg sift_s.knng sift_s.ssg 200 200 12 10 100 100 50 60 50 10 20 1.2
./test_query_shift ../data/gist/gist_base.fvecs gist.ssg gist_s.knng gist_s.ssg 400 400 12 15 100 500 70 60 10 10 50 1.2
./test_query_shift ../data/crawl/crawl_base.fvecs crawl.ssg crawl_s.knng crawl_s.ssg 400 420 12 15 100 500 40 60 2 10 40 1.2
./test_query_shift ../data/glove-100/glove-100_base.fvecs glove.ssg glove_s.knng glove_s.ssg 400 420 12 20 200 500 50 60 2 10 110 1.2