# =============================================================================
# ORB_SLAM3_Remastered
#
# Configure and build inside the dev container, never on the host:
#   ./docker/run.sh -- bash -c '
#     cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=Release &&
#     cmake --build /workspace/build -j 12'
#
# The SLAM library itself is not wired up yet -- see "Status" in README.md.
# What builds today is the wrapper layer over the pinned upstream dependencies,
# plus the tests and smoke test that prove the environment is sound.
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(ORB_SLAM3_Remastered VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Where the thirdparty image installs every pinned dependency.
list(APPEND CMAKE_PREFIX_PATH "$ENV{ORBSLAM3R_PREFIX}" /opt/orbslam3r)

if(NOT EXISTS /opt/orbslam3r/THIRDPARTY_MANIFEST.txt)
  message(WARNING
    "/opt/orbslam3r not found. This project expects to be configured inside the "
    "orbslam3r/dev container; run ./docker/run.sh first.")
endif()


# Must come before any add_subdirectory(), or add_test() calls in the
# subdirectories are silently dropped and ctest reports "No tests were found".
enable_testing()

add_subdirectory(vendor_ext)

option(ORBSLAM3R_BUILD_SMOKE_TEST "Build the dependency smoke test" ON)
if(ORBSLAM3R_BUILD_SMOKE_TEST)
  add_subdirectory(tools/smoke_test)
endif()
