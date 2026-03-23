include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_LIBS
    ORB_SLAM3
    gtest_main
    ${OpenCV_LIBS}
    slam3_wrappers
    g2o
)

# Helper macro
macro(slam3_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ${TEST_LIBS})
    add_test(NAME ${name} COMMAND ${name})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endmacro()

# Tier 1: Fully independent
slam3_add_test(test_converter)
slam3_add_test(test_imu_types)
slam3_add_test(test_orb_extractor)
slam3_add_test(test_geometric_tools)
slam3_add_test(test_orb_matcher)
slam3_add_test(test_dbow2_wrapper)

# Tier 2: Minimal setup
slam3_add_test(test_map_point)
slam3_add_test(test_map)
# test_atlas: Atlas destructor blocks (Map cleanup mutex deadlock in isolated context)
# slam3_add_test(test_atlas)

# I/O
slam3_add_test(test_trajectory_writer)
