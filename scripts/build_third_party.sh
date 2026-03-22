#!/bin/bash
# =============================================================================
# Build third-party submodules for ORB_SLAM3_Remastered
# Order: Pangolin -> DLib -> DBoW2 -> OpenGV
# (Sophus is header-only, g2o is built via add_subdirectory in CMakeLists.txt)
#
# Build directories are created in /tmp to keep source trees clean.
# All libraries are installed to /usr/local.
# =============================================================================
set -e

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}
BUILD_ROOT=/tmp/third_party_build
NPROC=$(nproc)

echo "=== Building third-party libraries (${NPROC} jobs) ==="
echo "    Source: ${WORKSPACE}/third_party"
echo "    Build:  ${BUILD_ROOT}"

# 1. Pangolin
echo ">>> [1/4] Building Pangolin..."
mkdir -p "${BUILD_ROOT}/Pangolin" && cd "${BUILD_ROOT}/Pangolin"
cmake "${WORKSPACE}/third_party/Pangolin" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PANGOLIN_LIBOPENEXR=OFF \
    > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
make install > /dev/null 2>&1
ldconfig
echo ">>> Pangolin installed"

# 2. DLib
echo ">>> [2/4] Building DLib..."
mkdir -p "${BUILD_ROOT}/DLib" && cd "${BUILD_ROOT}/DLib"
cmake "${WORKSPACE}/third_party/DLib" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
make install > /dev/null 2>&1
ldconfig
echo ">>> DLib installed"

# 3. DBoW2
echo ">>> [3/4] Building DBoW2..."
mkdir -p "${BUILD_ROOT}/DBoW2" && cd "${BUILD_ROOT}/DBoW2"
cmake "${WORKSPACE}/third_party/DBoW2" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
make install > /dev/null 2>&1
ldconfig
echo ">>> DBoW2 installed"

# 4. OpenGV
echo ">>> [4/4] Building OpenGV..."
mkdir -p "${BUILD_ROOT}/OpenGV" && cd "${BUILD_ROOT}/OpenGV"
cmake "${WORKSPACE}/third_party/OpenGV" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TESTS=OFF \
    > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
make install > /dev/null 2>&1
ldconfig
echo ">>> OpenGV installed"

echo "=== All third-party libraries built and installed to /usr/local ==="
