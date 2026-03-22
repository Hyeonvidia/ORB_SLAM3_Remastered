#!/bin/bash
# =============================================================================
# Build ORB_SLAM3_Remastered main library and examples
# =============================================================================
set -e

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}
NPROC=$(nproc)

echo "=== Building ORB_SLAM3_Remastered ==="

# Decompress vocabulary if needed
if [ -f "${WORKSPACE}/Vocabulary/ORBvoc.txt.tar.gz" ] && [ ! -f "${WORKSPACE}/Vocabulary/ORBvoc.txt" ]; then
    echo ">>> Decompressing vocabulary..."
    cd "${WORKSPACE}/Vocabulary"
    tar xf ORBvoc.txt.tar.gz
fi

cd "${WORKSPACE}"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j${NPROC}

echo "=== BUILD SUCCESS ==="
echo "Library: ${WORKSPACE}/lib/"
echo "Binaries: ${WORKSPACE}/bin/"
