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
# =============================================================================
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
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
#   SHARED <target>)             the part that is the shared library itself
function(orbslam3r_part name)
  cmake_parse_arguments(P "" "SHARED" "LAYERS;SOURCES;EXCEPT;USES;PACKAGES;HEADER_ONLY" ${ARGN})
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

  if(P_SHARED)
    set(target ${P_SHARED})
    add_library(${target} SHARED ${sources})
  else()
    set(target orbslam3r_${name})
    add_library(${target} OBJECT ${sources})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
  target_compile_features(${target} PUBLIC cxx_std_17)
  # Sophus's checks abort the process with their message on stdout, which a
  # file swallows; src/common/SophusEnsure.cpp puts it on stderr. PUBLIC,
  # because the check is inline in Sophus's headers and every user must see
  # the same one.
  target_compile_definitions(${target} PUBLIC SOPHUS_ENABLE_ENSURE_HANDLER)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  set(uses "")
  foreach(part ${P_USES})
    list(APPEND uses orbslam3r_${part})
  endforeach()
  target_link_libraries(${target} PUBLIC ${uses} ${P_PACKAGES})

  string(REPLACE ";" " " layers "${P_LAYERS}")
  string(REPLACE ";" " " files "${relative}")
  string(REPLACE ";" " " used "${P_USES}")
  string(REPLACE ";" " " packages "${P_PACKAGES} ${P_HEADER_ONLY}")
  set_property(GLOBAL APPEND_STRING PROPERTY ORBSLAM3R_PART_GRAPH
               "${name}|${layers}|${files}|${used}|${packages}\n")
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
           orbslam3r::g2o_ext     # Converter's g2o overloads
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

orbslam3r_part(slam
  LAYERS atlas tracking optimization local_mapping loop_closing
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/Settings.cpp
  USES common camera features
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} Boost::serialization OpenSSL::Crypto
           orbslam3r::dbow2_ext   # DBoW2 + the text vocabulary + serialization
           orbslam3r::g2o_ext     # g2o + the renamed-symbol aliases + solver factory
           # Tracking.cpp includes the viewer's drawers, and through them
           # Pangolin's headers, which do not compile without the definitions
           # its targets carry. Goes when Tracking reaches the drawers through
           # ports.
           ${Pangolin_LIBRARIES}
  HEADER_ONLY Sophus)

orbslam3r_part(viewer
  LAYERS viewer
  USES slam common
  PACKAGES Eigen3::Eigen ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  HEADER_ONLY Sophus)

orbslam3r_part(system SHARED ORB_SLAM3
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/System.cpp
  USES slam viewer common camera
  PACKAGES ${OpenCV_LIBS} Boost::serialization OpenSSL::Crypto ${Pangolin_LIBRARIES}
  HEADER_ONLY Sophus)
add_library(orbslam3r::ORB_SLAM3 ALIAS ORB_SLAM3)
# An object library's objects go only into what links it directly, not into
# what links something that links it; System.cpp includes nothing of features,
# and the library needs its objects all the same.
target_link_libraries(ORB_SLAM3 PUBLIC orbslam3r_features)

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
