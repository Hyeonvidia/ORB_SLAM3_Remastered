cmake_minimum_required(VERSION 3.16)
project(display_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(display_test display_test.cpp)
target_link_libraries(display_test
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)
target_include_directories(display_test PRIVATE
    ${Pangolin_INCLUDE_DIRS}
)
