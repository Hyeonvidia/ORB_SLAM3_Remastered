# =============================================================================
# Dataset example binaries.
#
# The RealSense examples upstream ships are not built here: they need
# librealsense2, which has no linux/arm64 package in this image and no camera to
# talk to from inside a container.
# =============================================================================
set(ORBSLAM3R_EXAMPLES
  Monocular/mono_euroc
  Monocular/mono_kitti
  Monocular/mono_tum
  Monocular/mono_tum_vi
  Monocular-Inertial/mono_inertial_euroc
  Monocular-Inertial/mono_inertial_tum_vi
  Stereo/stereo_euroc
  Stereo/stereo_kitti
  Stereo/stereo_tum_vi
  Stereo-Inertial/stereo_inertial_euroc
  Stereo-Inertial/stereo_inertial_tum_vi
  RGB-D/rgbd_tum
)

foreach(example ${ORBSLAM3R_EXAMPLES})
  get_filename_component(name ${example} NAME)
  get_filename_component(subdir ${example} DIRECTORY)
  add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cpp)
  target_link_libraries(${name} PRIVATE orbslam3r::ORB_SLAM3)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${subdir})
endforeach()
