cmake_minimum_required(VERSION 3.16)
project(ORB_SLAM3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(BUILD_TESTS "Build unit tests" ON)
option(WITH_GTSAM "Build with GTSAM optimizer backend" OFF)

# Compiler flags — NO -march=native for arm64 portability
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# Dependencies
find_package(OpenCV 4.4 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenSSL REQUIRED)
find_package(realsense2 QUIET)

# Third-party: g2o (out-of-source build) + wrappers
add_subdirectory(third_party/g2o ${CMAKE_BINARY_DIR}/g2o)
add_subdirectory(third_party/wrappers)

# Optional: GTSAM optimizer backend
if(WITH_GTSAM)
    find_package(GTSAM REQUIRED)
    message(STATUS "GTSAM: found (version ${GTSAM_VERSION})")
    add_definitions(-DWITH_GTSAM)
endif()

# Find installed third-party libraries (built by scripts/build_third_party.sh)
find_library(DBoW2_LIB DBoW2 PATHS /usr/local/lib REQUIRED)
find_library(DLib_LIB DLib PATHS /usr/local/lib REQUIRED)
message(STATUS "DBoW2: ${DBoW2_LIB}")
message(STATUS "DLib: ${DLib_LIB}")

# Include directories — modular structure
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/tracking
    ${PROJECT_SOURCE_DIR}/include/mapping
    ${PROJECT_SOURCE_DIR}/include/recognition
    ${PROJECT_SOURCE_DIR}/include/closing
    ${PROJECT_SOURCE_DIR}/include/atlas
    ${PROJECT_SOURCE_DIR}/include/visualization
    ${PROJECT_SOURCE_DIR}/include/camera
    ${PROJECT_SOURCE_DIR}/include/io
    ${PROJECT_SOURCE_DIR}/third_party/Sophus
    ${PROJECT_SOURCE_DIR}/third_party/g2o
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/include
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/include/DBoW2
    ${PROJECT_SOURCE_DIR}/third_party/DLib/include
    ${PROJECT_SOURCE_DIR}/third_party/wrappers/dbow2_wrapper
    ${PROJECT_SOURCE_DIR}/third_party/wrappers/opengv_wrapper
    ${PROJECT_SOURCE_DIR}/third_party/wrappers/g2o_wrapper
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
)

# Output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Main library — modular source layout
add_library(${PROJECT_NAME} SHARED
    # System orchestrator
    src/System.cpp
    # core/
    src/core/Frame.cpp
    src/core/KeyFrame.cpp
    src/core/MapPoint.cpp
    src/core/Map.cpp
    src/core/ImuTypes.cpp
    src/core/Converter.cpp
    src/core/GeometricTools.cpp
    src/core/Config.cpp
    src/core/Sim3Type.cpp
    # tracking/
    src/tracking/Tracking.cpp
    src/tracking/ORBextractor.cpp
    src/tracking/FeatureExtractor.cpp
    src/tracking/ORBmatcher.cpp
    src/tracking/MLPnPsolver.cpp
    # mapping/
    src/mapping/LocalMapping.cpp
    src/mapping/Optimizer.cpp
    src/mapping/G2oOptimizer.cpp
    src/mapping/G2oTypes.cpp
    src/mapping/OptimizableTypes.cpp
    src/mapping/TwoViewReconstruction.cpp
    # recognition/
    src/recognition/KeyFrameDatabase.cpp
    src/recognition/PlaceRecognition.cpp
    # closing/
    src/closing/LoopClosing.cpp
    src/closing/Sim3Solver.cpp
    # atlas/
    src/atlas/Atlas.cpp
    # visualization/
    src/visualization/Viewer.cpp
    src/visualization/FrameDrawer.cpp
    src/visualization/MapDrawer.cpp
    # camera/
    src/camera/Pinhole.cpp
    src/camera/KannalaBrandt8.cpp
    # io/
    src/io/Settings.cpp
    src/io/TrajectoryWriter.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    slam3_wrappers
    ${DBoW2_LIB}
    ${DLib_LIB}
    g2o
    Boost::serialization
    OpenSSL::Crypto
)

if(realsense2_FOUND)
    include_directories(${realsense_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
endif()

if(WITH_GTSAM)
    target_sources(${PROJECT_NAME} PRIVATE
        src/mapping/GtsamOptimizer.cpp
        src/mapping/ABTestOptimizer.cpp)
    target_link_libraries(${PROJECT_NAME} gtsam)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GTSAM_INCLUDE_DIR})
endif()

# =============================================================================
# Examples
# =============================================================================

# RGB-D
add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cpp)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# Stereo
add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc Examples/Stereo/stereo_euroc.cpp)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi Examples/Stereo/stereo_tum_vi.cpp)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

# Monocular
add_executable(mono_tum Examples/Monocular/mono_tum.cpp)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti Examples/Monocular/mono_kitti.cpp)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc Examples/Monocular/mono_euroc.cpp)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi Examples/Monocular/mono_tum_vi.cpp)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

# Monocular-Inertial
add_executable(mono_inertial_euroc Examples/Monocular-Inertial/mono_inertial_euroc.cpp)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi Examples/Monocular-Inertial/mono_inertial_tum_vi.cpp)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

# Stereo-Inertial
add_executable(stereo_inertial_euroc Examples/Stereo-Inertial/stereo_inertial_euroc.cpp)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi Examples/Stereo-Inertial/stereo_inertial_tum_vi.cpp)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

# RealSense examples (optional)
if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i Examples/RGB-D/rgbd_realsense_D435i.cpp)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})

    add_executable(rgbd_inertial_realsense_D435i Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cpp)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})

    add_executable(stereo_realsense_t265 Examples/Stereo/stereo_realsense_t265.cpp)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i Examples/Stereo/stereo_realsense_D435i.cpp)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})

    add_executable(mono_realsense_t265 Examples/Monocular/mono_realsense_t265.cpp)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i Examples/Monocular/mono_realsense_D435i.cpp)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_t265 Examples/Monocular-Inertial/mono_inertial_realsense_t265.cpp)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cpp)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_t265 Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cpp)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cpp)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_D435i Examples/Calibration/recorder_realsense_D435i.cpp)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265 Examples/Calibration/recorder_realsense_T265.cpp)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
