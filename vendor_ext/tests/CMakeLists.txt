find_package(OpenCV REQUIRED)

add_executable(test_vendor_ext test_vendor_ext.cpp)
target_link_libraries(test_vendor_ext PRIVATE
  orbslam3r::dbow2_ext
  orbslam3r::g2o_ext
  ${OpenCV_LIBS}
)
target_compile_features(test_vendor_ext PRIVATE cxx_std_17)

add_test(NAME vendor_ext COMMAND test_vendor_ext)
