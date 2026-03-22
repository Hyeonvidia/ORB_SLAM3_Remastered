add_library(slam3_dbow2_wrapper STATIC dbow2_wrapper.cpp)
set_target_properties(slam3_dbow2_wrapper PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(slam3_dbow2_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/include
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/include/DBoW2
    ${PROJECT_SOURCE_DIR}/third_party/DLib/include
)

target_link_libraries(slam3_dbow2_wrapper PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/build/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/third_party/DLib/build/libDLib.so
    ${OpenCV_LIBS}
    Boost::serialization
)
