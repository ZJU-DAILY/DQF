cd build
cmake ..
make
cd tests

./test_build_full_index ../data/sift/sift_base.fvecs ./sift.knng ./sift.ssg 200 200 12 10 100 100 50 60