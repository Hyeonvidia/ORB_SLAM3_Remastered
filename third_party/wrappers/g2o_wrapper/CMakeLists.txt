add_library(slam3_g2o_wrapper INTERFACE)

target_include_directories(slam3_g2o_wrapper INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/g2o
)
