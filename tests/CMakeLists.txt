set(CMAKE_CXX_STANDARD 11)

add_executable(test_query_shift test_query_shift.cpp)
target_link_libraries(test_query_shift ${OpenCV_LIBS})
target_link_libraries(test_query_shift ${PROJECT_NAME})

add_executable(test_ablation test_ablation.cpp)
target_link_libraries(test_ablation ${OpenCV_LIBS})
target_link_libraries(test_ablation ${PROJECT_NAME})
