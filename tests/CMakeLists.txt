set(CMAKE_CXX_STANDARD 11)

add_executable(test_create_dataset test_create_dataset.cpp)
target_link_libraries(test_create_dataset ${PROJECT_NAME})

# add_executable(test_full_gqi test_full_gqi.cpp)
# target_link_libraries(test_full_gqi ${OpenCV_LIBS})
# target_link_libraries(test_full_gqi ${PROJECT_NAME})

add_executable(test_query_shift test_query_shift.cpp)
target_link_libraries(test_query_shift ${OpenCV_LIBS})
target_link_libraries(test_query_shift ${PROJECT_NAME})

add_executable(test_ablation test_ablation.cpp)
target_link_libraries(test_ablation ${OpenCV_LIBS})
target_link_libraries(test_ablation ${PROJECT_NAME})

# add_executable(test_Dtree Atests/test_Dtree.cpp)
# target_link_libraries(test_Dtree ${OpenCV_LIBS})
# target_link_libraries(test_Dtree ${PROJECT_NAME})

# add_executable(test_Rtree Atests/test_Rtree.cpp)
# target_link_libraries(test_Rtree ${OpenCV_LIBS})
# target_link_libraries(test_Rtree ${PROJECT_NAME})


# add_executable(test_depth Atests/test_depth.cpp)
# target_link_libraries(test_depth ${OpenCV_LIBS})
# target_link_libraries(test_depth ${PROJECT_NAME})

# add_executable(test_step Atests/test_step.cpp)
# target_link_libraries(test_step ${OpenCV_LIBS})
# target_link_libraries(test_step ${PROJECT_NAME})

# add_executable(test_beta Atests/test_beta.cpp)
# target_link_libraries(test_beta ${OpenCV_LIBS})
# target_link_libraries(test_beta ${PROJECT_NAME})

# add_executable(test_K Atests/test_K.cpp)
# target_link_libraries(test_K ${OpenCV_LIBS})
# target_link_libraries(test_K ${PROJECT_NAME})

# add_executable(test_one_search Atests/test_one_search.cpp)
# target_link_libraries(test_one_search ${OpenCV_LIBS})
# target_link_libraries(test_one_search ${PROJECT_NAME})

# add_executable(test_add_step Atests/test_add_step.cpp)
# target_link_libraries(test_add_step ${OpenCV_LIBS})
# target_link_libraries(test_add_step ${PROJECT_NAME})

# add_executable(test_query_graph Atests/test_query_graph.cpp)
# target_link_libraries(test_query_graph ${OpenCV_LIBS})
# target_link_libraries(test_query_graph ${PROJECT_NAME})