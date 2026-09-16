#!/usr/bin/env bash
# Regenerates src/ and include/ from the pristine ORB-SLAM3 v1.0 checkout.
#
# The port is three scripted stages, so it can be replayed at any time -- after
# re-pinning a submodule, or to audit exactly what the remaster changes:
#
#   1. port_body.py    include graph -> pinned upstream + vendor_ext wrappers,
#                      and .cc -> .cpp
#   2. qualify_std.py  std:: qualification + the standard headers the code had
#                      been getting through a leaked `using namespace std`
#   3. port_fixes.py   targeted source fixes the first two cannot express
#
# WARNING: this discards any hand edit in src/ or include/.  Put changes in
# tools/port_fixes.py instead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF=reference/ORB_SLAM3
[ -d "$REF/src" ] || { echo "missing $REF -- run tools/init_submodules.sh"; exit 1; }

if [ "${1:-}" != "--force" ]; then
  echo "This rewrites src/ and include/ from $REF, discarding hand edits."
  echo "Re-run with --force to proceed."
  exit 1
fi

# Keep the build definition; everything else comes from the reference.
CMAKE_BACKUP="$(mktemp)"
cp src/CMakeLists.txt "$CMAKE_BACKUP"

EXAMPLES_CMAKE_BACKUP="$(mktemp)"
cp Examples/CMakeLists.txt "$EXAMPLES_CMAKE_BACKUP" 2>/dev/null || : > "$EXAMPLES_CMAKE_BACKUP"

rm -rf src include Examples
mkdir -p src
cp -R "$REF/include" include
cp "$REF"/src/*.cc "$REF"/src/*.cpp src/
cp -R "$REF/src/CameraModels" src/CameraModels
cp "$CMAKE_BACKUP" src/CMakeLists.txt
rm -f "$CMAKE_BACKUP"

# Dataset-driven examples only.  The RealSense ones need librealsense2, which
# has no linux/arm64 package here and no camera attached to the container.
mkdir -p Examples
for d in Monocular Monocular-Inertial Stereo Stereo-Inertial RGB-D; do
  mkdir -p "Examples/$d"
  cp -R "$REF/Examples/$d"/* "Examples/$d/" 2>/dev/null || true
done
rm -f Examples/*/*realsense* Examples/*/RealSense*.yaml
if [ -s "$EXAMPLES_CMAKE_BACKUP" ]; then
  cp "$EXAMPLES_CMAKE_BACKUP" Examples/CMakeLists.txt
fi
rm -f "$EXAMPLES_CMAKE_BACKUP"

echo "== stage 1: include graph"
./tools/port_body.py
echo
echo "== stage 2: std:: qualification"
./tools/qualify_std.py
echo
echo "== stage 3: targeted fixes"
./tools/port_fixes.py
