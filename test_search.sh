cd build
cmake ..
make
cd tests

./test_search ../data/sift/sift_base.fvecs sift.ssg sift_s.knng sift_s.ssg 200 200 12 10 100 100 50 60 50 10 1.2
./test_search ../data/gist/gist_base.fvecs gist.ssg gist_s.knng gist_s.ssg 400 400 12 15 100 500 70 60 10 10 1.2
./test_search ../data/crawl/crawl_base.fvecs crawl.ssg crawl_s.knng crawl_s.ssg 400 420 12 15 100 500 40 60 2 10 1.2
./test_search ../data/glove-100/glove-100_base.fvecs glove.ssg glove_s.knng glove_s.ssg 400 420 12 20 200 500 50 60 2 10 1.2