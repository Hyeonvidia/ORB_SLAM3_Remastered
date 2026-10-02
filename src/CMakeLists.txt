# =============================================================================
# libORB_SLAM3 — the SLAM library itself.
#
# Sources are ORB-SLAM3 v1.0's, with their include graph rewritten onto the
# pinned upstream packages and the vendor_ext wrappers by tools/port_body.py,
# and std:: qualification restored by tools/qualify_std.py.
#
# The library is one shared object, as before, but it is put together from
# parts that each say what they are made of and what they use. A part may
# include the headers of the parts and the packages it names here and of no
# others; tools/deps_check.py holds every #include to that, and ctest runs it.
# The parts are object libraries rather than static ones so that each object
# goes into the shared library exactly once and nothing is dropped for not
# being referenced.
#
#   -DORBSLAM3R_BUILD_VIEWER=OFF   leaves the viewer out: Pangolin and OpenGL
#                                  are then neither looked for nor linked
# =============================================================================
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

set_property(GLOBAL PROPERTY ORBSLAM3R_PART_GRAPH "") # what tools/deps_check.py reads

# orbslam3r_part(<name>
#   LAYERS  <folder>...          folders of src/ and include/ that are this part's
#   SOURCES <file>...            sources from other folders that are this part's
#   EXCEPT  <file>...            sources of LAYERS that are another part's
#   USES    <part>...            parts whose headers this one may include
#   PACKAGES <target>...         packages it links and may include
#   HEADER_ONLY <name>...        packages it may include that have no target
#   NOT_BUILT)                   said, and checked, but left out of this build
function(orbslam3r_part name)
  cmake_parse_arguments(P "NOT_BUILT" "" "LAYERS;SOURCES;EXCEPT;USES;PACKAGES;HEADER_ONLY" ${ARGN})
  set(sources ${P_SOURCES})
  foreach(layer ${P_LAYERS})
    file(GLOB layer_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${layer}/*.cpp)
    list(APPEND sources ${layer_sources})
  endforeach()
  foreach(other ${P_EXCEPT})
    list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/${other})
  endforeach()
  set(relative "")
  foreach(source ${sources})
    file(RELATIVE_PATH rel ${CMAKE_CURRENT_SOURCE_DIR} ${source})
    list(APPEND relative ${rel})
  endforeach()

  string(REPLACE ";" " " layers "${P_LAYERS}")
  string(REPLACE ";" " " files "${relative}")
  string(REPLACE ";" " " used "${P_USES}")
  string(REPLACE ";" " " packages "${P_PACKAGES} ${P_HEADER_ONLY}")
  set_property(GLOBAL APPEND_STRING PROPERTY ORBSLAM3R_PART_GRAPH
               "${name}|${layers}|${files}|${used}|${packages}\n")
  if(P_NOT_BUILT)
    return()
  endif()

  set(target orbslam3r_${name})
  if(sources)
    set(scope PUBLIC)
    add_library(${target} OBJECT ${sources})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  else()
    set(scope INTERFACE) # a part that is headers only, so far
    add_library(${target} INTERFACE)
  endif()
  target_compile_features(${target} ${scope} cxx_std_17)
  # Sophus's checks abort the process with their message on stdout, which a
  # file swallows; src/common/SophusEnsure.cpp puts it on stderr. PUBLIC,
  # because the check is inline in Sophus's headers and every user must see
  # the same one.
  target_compile_definitions(${target} ${scope} SOPHUS_ENABLE_ENSURE_HANDLER)
  target_include_directories(${target} ${scope}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  set(uses "")
  foreach(part ${P_USES})
    list(APPEND uses orbslam3r_${part})
  endforeach()
  target_link_libraries(${target} ${scope} ${uses} ${P_PACKAGES})
endfunction()

# One folder per layer of the architecture in Figure 1 of the ORB-SLAM3 paper;
# see tools/layer_layout.py for what belongs where. Three of the layers stand
# on their own today. The rest know each other in both directions -- the atlas
# holds Frames, tracking calls the optimiser and the optimiser reads keyframes
# -- and are one part until those are undone.
orbslam3r_part(common
  LAYERS common
  EXCEPT common/Settings.cpp   # builds the cameras, so it is above them
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} Boost::serialization
           orbslam3r::dbow2_ext   # DUtils::Random, in TwoViewReconstruction's RANSAC
  HEADER_ONLY Sophus)

orbslam3r_part(camera
  LAYERS camera
  USES common
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} Boost::serialization
  HEADER_ONLY Sophus)

orbslam3r_part(features
  LAYERS features
  USES common
  PACKAGES ${OpenCV_LIBS})

# An optimisation said without the map and without the library that solves it:
# what is to be found, from what, and what a solver of it must do.
orbslam3r_part(optim
  LAYERS optim
  PACKAGES Eigen3::Eigen)

# Those solvers, on g2o, and the g2o edges they and the optimisations not yet
# taken apart are built of.
orbslam3r_part(optim_g2o
  LAYERS optim_g2o
  USES optim camera common   # common: the preintegration of the inertial edges
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} orbslam3r::g2o_ext)

orbslam3r_part(slam
  LAYERS atlas tracking optimization local_mapping loop_closing
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/Settings.cpp
  USES common camera features optim optim_g2o
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} Boost::serialization OpenSSL::Crypto
           orbslam3r::dbow2_ext   # DBoW2 + the text vocabulary + serialization
           orbslam3r::g2o_ext     # g2o + the renamed-symbol aliases + solver factory
  HEADER_ONLY Sophus)

orbslam3r_part(system
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/System.cpp
  USES slam common camera
  PACKAGES ${OpenCV_LIBS} Boost::serialization OpenSSL::Crypto
  HEADER_ONLY Sophus)

# The viewer is above System: its buttons call System, and System knows it
# only as the port and the three functions of include/ViewerPort.hpp. Whichever
# of the two parts below is built defines those functions.
option(ORBSLAM3R_BUILD_VIEWER "Build the Pangolin viewer into the library" ON)
if(ORBSLAM3R_BUILD_VIEWER)
  find_package(Pangolin REQUIRED)
  set(with_viewer "")
  set(without_viewer NOT_BUILT)
else()
  set(with_viewer NOT_BUILT)
  set(without_viewer "")
endif()
orbslam3r_part(viewer ${with_viewer}
  LAYERS viewer
  USES system slam common
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  HEADER_ONLY Sophus pangolin)
orbslam3r_part(noviewer ${without_viewer}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/NoViewer.cpp
  USES system slam
  HEADER_ONLY Sophus)

# -DORBSLAM3R_OPT_SHADOW=ON: every optimisation that has been taken apart runs
# twice on the same input, as it is now and as v1.0 wrote it, and the calls
# whose results differ in any bit are counted (optimization/Shadow.hpp). The
# two are different code computing the same expressions, and only without
# fused multiply-add do the same expressions give the same bits whatever the
# compiler inlined: with it, a product is rounded or not according to what it
# was inlined next to. So the flag, on what the two are compiled from.
option(ORBSLAM3R_OPT_SHADOW "Run taken-apart optimisations beside v1.0's and count the calls that differ" OFF)
if(ORBSLAM3R_OPT_SHADOW)
  foreach(part slam optim_g2o)
    target_compile_definitions(orbslam3r_${part} PRIVATE ORBSLAM3R_OPT_SHADOW)
    target_compile_options(orbslam3r_${part} PRIVATE -ffp-contract=off)
  endforeach()
endif()

# A shared library with no sources of its own. The parts are named one by one:
# an object library's objects go only into what links it directly, not into
# what links something that links it.
add_library(ORB_SLAM3 SHARED)
add_library(orbslam3r::ORB_SLAM3 ALIAS ORB_SLAM3)
target_link_libraries(ORB_SLAM3 PUBLIC
  orbslam3r_common
  orbslam3r_camera
  orbslam3r_features
  orbslam3r_optim
  orbslam3r_optim_g2o
  orbslam3r_slam
  orbslam3r_system
  $<IF:$<BOOL:${ORBSLAM3R_BUILD_VIEWER}>,orbslam3r_viewer,orbslam3r_noviewer>
)

get_property(part_graph GLOBAL PROPERTY ORBSLAM3R_PART_GRAPH)
file(WRITE ${CMAKE_BINARY_DIR}/part_graph.txt "${part_graph}")

# A descriptor's bits are comparisons of pixels whose coordinates are rounded
# sums of products. Where the compiler may fuse a product into the sum, as it
# does by default on ARM, some of them round the other way, and the same image
# gives other descriptors than on x86. Not in this file.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/features/ORBextractor.cpp
                            PROPERTIES COMPILE_OPTIONS -ffp-contract=off)

set_target_properties(ORB_SLAM3 PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
