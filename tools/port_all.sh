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
# THE TREE IS NOW HAND-MAINTAINED.
#   This pipeline was the bootstrap: it derived src/, include/ and Examples/
#   from the pristine reference and recorded, as three readable scripts, exactly
#   what the remaster changes. That job is done. Refactoring is structural work
#   -- moving types, splitting files, changing ownership -- which cannot be
#   expressed as string replacements, so from here the tree is edited directly
#   and this script would destroy that work.
#
#   It is kept because it is the record of how the tree was derived, and because
#   re-deriving from a different upstream pin may one day be worth doing. It now
#   refuses to run without --bootstrap, on top of --force.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF=reference/ORB_SLAM3
[ -d "$REF/src" ] || { echo "missing $REF -- run tools/init_submodules.sh"; exit 1; }

want_bootstrap=0
want_force=0
for arg in "$@"; do
  case "$arg" in
    --bootstrap) want_bootstrap=1 ;;
    --force)     want_force=1 ;;
  esac
done

if [ "$want_force" != 1 ] || [ "$want_bootstrap" != 1 ]; then
  cat >&2 <<'MSG'
src/, include/ and Examples/ are hand-maintained now -- this pipeline was the
one-time bootstrap that derived them, and re-running it would discard every
refactoring since.

Read the header of this file. If you really mean to re-derive the whole tree
from reference/ORB_SLAM3, discarding local work:

    ./tools/port_all.sh --bootstrap --force
MSG
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
