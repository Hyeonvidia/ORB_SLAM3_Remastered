# Compiles and links against every dependency baked into orbslam3r/thirdparty.
# Its job is to fail loudly at image-build time rather than halfway through a
# SLAM run, and to serve as the reference for how the modern g2o API is called.
cmake_minimum_required(VERSION 3.16)
project(orbslam3r_smoke_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
list(APPEND CMAKE_PREFIX_PATH "$ENV{ORBSLAM3R_PREFIX}" /opt/orbslam3r)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus QUIET)

add_executable(smoke_test main.cpp)
target_link_libraries(smoke_test PRIVATE
  Eigen3::Eigen
  ${OpenCV_LIBS}
  DBoW2::DBoW2
  g2o::core g2o::stuff g2o::types_sba g2o::types_sim3 g2o::solver_eigen
  ${Pangolin_LIBRARIES}
)
# Pangolin 0.9 exports no version macro from its headers, so take the version
# CMake already knows and hand it to the source.
target_compile_definitions(smoke_test PRIVATE
  ORBSLAM3R_PANGOLIN_VERSION="${Pangolin_VERSION}")

if(Sophus_FOUND)
  target_link_libraries(smoke_test PRIVATE Sophus::Sophus)
else()
  target_include_directories(smoke_test PRIVATE /opt/orbslam3r/include)
endif()

# Registered without --gl: ctest must pass on a machine with no display at all.
add_test(NAME dependency_smoke COMMAND smoke_test)
