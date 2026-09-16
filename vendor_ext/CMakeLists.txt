# =============================================================================
# vendor_ext — every change ORB-SLAM3 made to a borrowed library, expressed as
# code that sits BESIDE the pristine upstream instead of inside it.
#
#   dbow2_build/  build definition for the upstream DBoW2 subset we compile
#                 (upstream's own CMakeLists cannot be used; see that file)
#   dbow2_ext/    the text vocabulary format + non-intrusive serialization
#   g2o_ext/      renamed-symbol aliases + the modern solver-ownership helper
#
# Nothing here edits thirdparty/.  Run tools/classify_vendored.sh at any time to
# confirm those checkouts are still byte-identical to upstream.
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# Declared so this directory also configures standalone, not only via the
# top-level project.
if(NOT PROJECT_NAME)
  project(orbslam3r_vendor_ext LANGUAGES CXX)
endif()

find_package(Boost REQUIRED COMPONENTS serialization)
find_package(DBoW2 REQUIRED)
find_package(g2o REQUIRED)

add_subdirectory(dbow2_ext)
add_subdirectory(g2o_ext)

option(ORBSLAM3R_BUILD_TESTS "Build the vendor_ext wrapper tests" ON)
if(ORBSLAM3R_BUILD_TESTS)
  add_subdirectory(tests)
endif()
