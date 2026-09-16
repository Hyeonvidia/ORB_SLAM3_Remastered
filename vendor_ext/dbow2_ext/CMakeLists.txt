# Header-only extensions over pristine upstream DBoW2.
add_library(orbslam3r_dbow2_ext INTERFACE)
add_library(orbslam3r::dbow2_ext ALIAS orbslam3r_dbow2_ext)

target_include_directories(orbslam3r_dbow2_ext INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(orbslam3r_dbow2_ext INTERFACE cxx_std_17)
target_link_libraries(orbslam3r_dbow2_ext INTERFACE
  DBoW2::DBoW2
  Boost::serialization
)
