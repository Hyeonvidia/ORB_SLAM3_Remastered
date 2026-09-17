#!/usr/bin/env bash
# Watch ONE sequence live from macOS.
#
#   ./tools/monitor.sh euroc stereo MH01
#   ./tools/monitor.sh euroc stereo-inertial V203
#   ./tools/monitor.sh kitti stereo 04
#   ./tools/monitor.sh tum rgbd
#
# A thin wrapper over tools/run_gui.sh, which runs a whole dataset the same way
# and carries the explanation of the X11 and VNC transports. This exists because
# a single sequence is the common case and this is the name that gets typed;
# everything else about it -- --vnc, --port, --no-open -- is passed straight
# through.
#
# One difference from run_gui.sh's own spelling: sensor configurations are
# written with hyphens here ("stereo-inertial") because that is what this script
# has always taken. They are translated below.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARGS=()
for a in "$@"; do
  case "$a" in
    mono-inertial|stereo-inertial) ARGS+=("${a//-/_}") ;;
    *)                             ARGS+=("$a") ;;
  esac
done

exec "$ROOT/tools/run_gui.sh" ${ARGS[@]+"${ARGS[@]}"}
