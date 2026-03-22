add_subdirectory(dbow2_wrapper)
add_subdirectory(opengv_wrapper)
add_subdirectory(g2o_wrapper)

# Combined interface target
add_library(slam3_wrappers INTERFACE)
target_link_libraries(slam3_wrappers INTERFACE
    slam3_dbow2_wrapper
    slam3_opengv_wrapper
    slam3_g2o_wrapper
)
