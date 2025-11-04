set(CMAKE_CXX_STANDARD 11)

add_executable(test_query_shift test_query_shift.cpp)
target_link_libraries(test_query_shift ${OpenCV_LIBS})
target_link_libraries(test_query_shift ${PROJECT_NAME})

add_executable(test_search test_search.cpp)
target_link_libraries(test_search ${OpenCV_LIBS})
target_link_libraries(test_search ${PROJECT_NAME})

add_executable(test_build_full_index test_build_full_index.cpp)
target_link_libraries(test_build_full_index ${OpenCV_LIBS})
target_link_libraries(test_build_full_index ${PROJECT_NAME})

