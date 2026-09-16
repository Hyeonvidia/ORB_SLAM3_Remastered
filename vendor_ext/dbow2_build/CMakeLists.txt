# =============================================================================
# Build definition for the subset of upstream DBoW2 that ORB-SLAM3 uses.
#
# WHY THIS FILE EXISTS
#   thirdparty/DBoW2 is a pristine upstream checkout and must stay that way, so
#   we cannot edit its CMakeLists.txt.  But upstream's own build unconditionally
#   compiles FBrief.cpp, which needs DLib's DVision library, whose .cpp files use
#   the OpenCV 1.x C API (IplImage / CvScalar / CV_RGB2GRAY) that OpenCV 4
#   removed.  ORB-SLAM3 never uses BRIEF descriptors.
#
#   So instead of patching upstream, we own the build definition and compile
#   exactly the translation units ORB-SLAM3 needs, straight from the untouched
#   upstream sources.
#
# NOTE ON DVision
#   FORB.cpp carries `#include <DVision/DVision.h>`, but references no DVision
#   symbol (verified: 0 occurrences of `DVision::`).  DLib's DVision *headers*
#   do compile against OpenCV 4 — only its .cpp files do not — so putting the
#   header directory on the include path satisfies the include with no library
#   to link against.
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(DBoW2Subset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Pristine upstream source trees, passed in by the caller.
set(DBOW2_UPSTREAM_DIR "" CACHE PATH "Path to the pristine upstream DBoW2 checkout")
set(DLIB_UPSTREAM_DIR  "" CACHE PATH "Path to the pristine upstream DLib checkout")
foreach(v DBOW2_UPSTREAM_DIR DLIB_UPSTREAM_DIR)
  if(NOT ${v} OR NOT EXISTS "${${v}}")
    message(FATAL_ERROR "${v} must point at an existing upstream checkout (got '${${v}}')")
  endif()
endforeach()

find_package(OpenCV REQUIRED COMPONENTS core)

# --- the subset -------------------------------------------------------------
# Excluded on purpose: FBrief.cpp (BRIEF descriptors, needs DVision library).
set(DBOW2_SOURCES
  ${DBOW2_UPSTREAM_DIR}/src/BowVector.cpp
  ${DBOW2_UPSTREAM_DIR}/src/FeatureVector.cpp
  ${DBOW2_UPSTREAM_DIR}/src/FORB.cpp
  ${DBOW2_UPSTREAM_DIR}/src/QueryResults.cpp
  ${DBOW2_UPSTREAM_DIR}/src/ScoringObject.cpp
)

add_library(DBoW2 SHARED ${DBOW2_SOURCES})
add_library(DBoW2::DBoW2 ALIAS DBoW2)

# Upstream's sources and headers include each other by bare name
# (`#include "FORB.h"`), so include/DBoW2 itself must be on the path for both
# this build and every consumer -- not just include/.
target_include_directories(DBoW2 PUBLIC
  $<BUILD_INTERFACE:${DBOW2_UPSTREAM_DIR}/include>
  $<BUILD_INTERFACE:${DBOW2_UPSTREAM_DIR}/include/DBoW2>
  $<BUILD_INTERFACE:${DLIB_UPSTREAM_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/DBoW2>
)
# TemplatedVocabulary::create() is a header template that calls
# DUtils::Random, so every consumer of DBoW2 needs DLib on the link line --
# hence PUBLIC, and hence found by path rather than by target (DLib ships an
# old-style Config module that exports variables, not an imported target).
find_library(DLIB_LIBRARY NAMES DLib REQUIRED
             HINTS ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_PREFIX_PATH}/lib)
target_link_libraries(DBoW2 PUBLIC ${OpenCV_LIBS} ${DLIB_LIBRARY})
set_target_properties(DBoW2 PROPERTIES VERSION 1.1.0 SOVERSION 1)

# --- install ----------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS DBoW2 EXPORT DBoW2Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${DBOW2_UPSTREAM_DIR}/include/DBoW2
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# DVision/DUtils headers travel with it: FORB.h/FORB.cpp include them.
install(DIRECTORY ${DLIB_UPSTREAM_DIR}/include/DVision
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# The generated DBoW2Config re-finds DLib, so a downstream project only has to
# find_package(DBoW2) to get a complete link line.
set(DBOW2_CONFIG_DLIB_HINT ${CMAKE_INSTALL_PREFIX}/lib)

install(EXPORT DBoW2Targets
        FILE DBoW2Targets.cmake
        NAMESPACE DBoW2::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DBoW2)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake
"include(CMakeFindDependencyMacro)\n\
find_dependency(OpenCV REQUIRED COMPONENTS core)\n\
find_library(DLIB_LIBRARY NAMES DLib REQUIRED HINTS ${DBOW2_CONFIG_DLIB_HINT})\n\
include(\"\${CMAKE_CURRENT_LIST_DIR}/DBoW2Targets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DBoW2)
