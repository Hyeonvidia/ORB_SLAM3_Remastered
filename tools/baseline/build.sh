#!/usr/bin/env bash
# Builds pristine ORB-SLAM3 v1.0 -- the reference/ORB_SLAM3 submodule -- in
# the dev container, so that the remaster can be measured against what it came
# from on the same machine, the same images and the same OpenCV.
#
#   ./tools/baseline/build.sh          # -> results/baseline/ORB_SLAM3/{lib,Examples}
#
# The copy is rebuilt from scratch each time; reference/ is never touched.
#
# v1.0 does not build or run here as it is, and v1.0-build.patch is the least
# that makes it: C++17 in place of C++11, because the pinned Pangolin needs
# it; mnFullBAIdx++ on a bool, which C++17 rejects; the Settings printer
# dereferencing a second-camera calibration that rectified stereo never sets,
# which crashes KITTI stereo before the first image; and the KITTI examples
# started without the viewer, so they can run unattended. None of it touches
# what the system computes.
#
# Its own Thirdparty/DBoW2 and g2o are built and used, as upstream's build.sh
# does. They must come first on the library path when the binaries run,
# because /opt/orbslam3r holds a DBoW2 with a different ABI; tools/ab_kitti.sh
# sets that up.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DST=results/baseline/ORB_SLAM3
rm -rf "$DST"
mkdir -p results/baseline
cp -R reference/ORB_SLAM3 "$DST"
rm -rf "$DST/.git"
patch -p1 -d "$DST" < tools/baseline/v1.0-build.patch

./docker/run.sh -- bash -c "
  set -e
  export CMAKE_PREFIX_PATH=/opt/orbslam3r
  cd /workspace/$DST
  for d in Thirdparty/DBoW2 Thirdparty/g2o; do
    mkdir -p \$d/build
    (cd \$d/build && cmake .. -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 && make -j12 > make.log 2>&1)
  done
  mkdir -p build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1
  make -j12 > make.log 2>&1
  echo \"errors=\$(grep -c 'error' make.log || true)\"
  ls ../lib/libORB_SLAM3.so ../Examples/Monocular/mono_kitti ../Examples/Stereo/stereo_kitti
"
