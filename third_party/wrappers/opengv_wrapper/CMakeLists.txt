add_library(slam3_opengv_wrapper STATIC opengv_wrapper.cpp)
set_target_properties(slam3_opengv_wrapper PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(slam3_opengv_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)
