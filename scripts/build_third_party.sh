#!/bin/bash
# =============================================================================
# Build third-party submodules for ORB_SLAM3_Remastered
# Order: Pangolin -> DLib -> DBoW2 -> g2o -> OpenGV
# (Sophus is header-only, no build needed)
# =============================================================================
set -e

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}
NPROC=$(nproc)

echo "=== Building third-party libraries (${NPROC} jobs) ==="

# 1. Pangolin
echo ">>> [1/5] Building Pangolin..."
cd "${WORKSPACE}/third_party/Pangolin"
rm -rf build && mkdir build && cd build
cmake .. \
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
echo ">>> [2/5] Building DLib..."
cd "${WORKSPACE}/third_party/DLib"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
echo ">>> DLib built"

# 3. DBoW2
echo ">>> [3/5] Building DBoW2..."
cd "${WORKSPACE}/third_party/DBoW2"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
echo ">>> DBoW2 built"

# 4. g2o
echo ">>> [4/5] Building g2o..."
cd "${WORKSPACE}/third_party/g2o"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
echo ">>> g2o built"

# 5. OpenGV
echo ">>> [5/5] Building OpenGV..."
cd "${WORKSPACE}/third_party/OpenGV"
rm -rf build && mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    > /dev/null 2>&1
make -j${NPROC} > /dev/null 2>&1
echo ">>> OpenGV built"

echo "=== All third-party libraries built successfully ==="
