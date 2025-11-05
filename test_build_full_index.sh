cd build
cmake ..
make
cd tests

./test_build_full_index ../data/sift/sift_base.fvecs ./sift.knng ./sift.ssg 200 200 12 10 100 100 50 60
./test_build_full_index ../data/gist/gist_base.fvecs ./gist.knng ./gist.ssg 400 400 12 15 100 500 70 60
./test_build_full_index ../data/crawl/crawl_base.fvecs ./crawl.knng ./crawl.ssg 400 420 12 15 100 500 40 60
./test_build_full_index ../data/glove-100/glove-100_base.fvecs ./glove.knng ./glove.ssg 400 420 12 20 200 500 50 60