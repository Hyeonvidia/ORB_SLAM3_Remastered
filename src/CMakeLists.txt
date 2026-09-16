# =============================================================================
# libORB_SLAM3 — the SLAM library itself.
#
# Sources are ORB-SLAM3 v1.0's, with their include graph rewritten onto the
# pinned upstream packages and the vendor_ext wrappers by tools/port_body.py,
# and std:: qualification restored by tools/qualify_std.py.
# =============================================================================
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

# One folder per layer of the architecture in Figure 1 of the ORB-SLAM3 paper;
# see tools/layer_layout.py for what belongs where.
file(GLOB_RECURSE ORBSLAM3_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(ORB_SLAM3 SHARED ${ORBSLAM3_SOURCES})
add_library(orbslam3r::ORB_SLAM3 ALIAS ORB_SLAM3)

target_include_directories(ORB_SLAM3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(ORB_SLAM3 PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  Boost::serialization
  OpenSSL::Crypto
  orbslam3r::dbow2_ext      # DBoW2 + the text vocabulary + serialization
  orbslam3r::g2o_ext        # g2o + the renamed-symbol aliases + solver factory
)

target_compile_features(ORB_SLAM3 PUBLIC cxx_std_17)
set_target_properties(ORB_SLAM3 PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
