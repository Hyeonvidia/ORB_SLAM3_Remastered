# Header-only compatibility layer over pristine upstream g2o.
add_library(orbslam3r_g2o_ext INTERFACE)
add_library(orbslam3r::g2o_ext ALIAS orbslam3r_g2o_ext)

target_include_directories(orbslam3r_g2o_ext INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(orbslam3r_g2o_ext INTERFACE cxx_std_17)
target_link_libraries(orbslam3r_g2o_ext INTERFACE
  g2o::core g2o::stuff g2o::types_sba g2o::types_sim3 g2o::types_slam3d
  g2o::solver_eigen g2o::solver_dense
)
