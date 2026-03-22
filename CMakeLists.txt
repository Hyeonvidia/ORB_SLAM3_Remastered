cmake_minimum_required(VERSION 3.16)
project(ORB_SLAM3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# Third-party: g2o (submodule)
add_subdirectory(third_party/g2o)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels
    ${PROJECT_SOURCE_DIR}/third_party/Sophus
    ${PROJECT_SOURCE_DIR}/third_party/g2o
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/include
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/include/DBoW2
    ${PROJECT_SOURCE_DIR}/third_party/DLib/include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
)

# Output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Main library
add_library(${PROJECT_NAME} SHARED
    src/System.cpp
    src/Tracking.cpp
    src/LocalMapping.cpp
    src/LoopClosing.cpp
    src/ORBextractor.cpp
    src/ORBmatcher.cpp
    src/FrameDrawer.cpp
    src/Converter.cpp
    src/MapPoint.cpp
    src/KeyFrame.cpp
    src/Atlas.cpp
    src/Map.cpp
    src/MapDrawer.cpp
    src/Optimizer.cpp
    src/Frame.cpp
    src/KeyFrameDatabase.cpp
    src/Sim3Solver.cpp
    src/Viewer.cpp
    src/ImuTypes.cpp
    src/G2oTypes.cpp
    src/CameraModels/Pinhole.cpp
    src/CameraModels/KannalaBrandt8.cpp
    src/OptimizableTypes.cpp
    src/MLPnPsolver.cpp
    src/GeometricTools.cpp
    src/TwoViewReconstruction.cpp
    src/Config.cpp
    src/Settings.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/build/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/third_party/DLib/build/libDLib.so
    g2o
    Boost::serialization
    OpenSSL::Crypto
)

if(realsense2_FOUND)
    include_directories(${realsense_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
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
