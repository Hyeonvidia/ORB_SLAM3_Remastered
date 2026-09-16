#!/usr/bin/env bash
# Watch the ORB-SLAM3 viewer live from macOS while a sequence runs.
#
#   ./tools/monitor.sh euroc stereo MH01
#   ./tools/monitor.sh euroc stereo-inertial V203
#   ./tools/monitor.sh kitti stereo 04
#   ./tools/monitor.sh tum rgbd
#
#   --no-open   do not launch Screen Sharing; just print the URL
#   --port N    use a different local port (default 5900)
#
# The VNC password defaults to "orbslam3r"; override with ORBSLAM3R_VNC_PASSWORD.
# It is not optional: macOS Screen Sharing never finishes the handshake against
# a server that offers only RFB security type 1 (None) -- it sits on
# "Connecting..." forever. Setting a password makes x11vnc offer type 2, VNC
# Authentication, which Apple's client does support.
#
# WHY VNC AND NOT X11
#   Forwarding X11 to XQuartz does not work for this viewer. XQuartz reaches the
#   container fine, but its indirect GLX exposes only OpenGL 1.4 with no direct
#   rendering, and Pangolin then fails to find a usable framebuffer config:
#
#       No matching fbConfigs or visuals found
#       glx: failed to create drisw screen
#       MESA: error: Failed to attach to x11 shm      (repeatedly)
#
#   That is a limit of XQuartz's GLX, not something the container can fix. So
#   rendering stays inside the container on Mesa's llvmpipe -- where the viewer
#   already works -- and x11vnc exports the Xvfb screen. Only pixels cross over,
#   and macOS has a VNC client built in.
#
#   The port is published on 127.0.0.1 only, so the screen is not reachable from
#   the network.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT=5900
OPEN=1
VNC_PASSWORD="${ORBSLAM3R_VNC_PASSWORD:-orbslam3r}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-open) OPEN=0; shift ;;
    --port)    PORT="$2"; shift 2 ;;
    *)         ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

DATASET="${1:-}"
CONFIG="${2:-}"
SEQ="${3:-}"
V=/workspace/Vocabulary/ORBvoc.txt
E=/workspace/Examples

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}
[ -n "$DATASET" ] && [ -n "$CONFIG" ] || usage

case "${DATASET}_${CONFIG}" in
  euroc_mono)            BIN=Monocular/mono_euroc;                   D=Monocular ;;
  euroc_stereo)          BIN=Stereo/stereo_euroc;                    D=Stereo ;;
  euroc_mono-inertial)   BIN=Monocular-Inertial/mono_inertial_euroc; D=Monocular-Inertial ;;
  euroc_stereo-inertial) BIN=Stereo-Inertial/stereo_inertial_euroc;  D=Stereo-Inertial ;;
  kitti_mono)            BIN=Monocular/mono_kitti;                   D=Monocular ;;
  kitti_stereo)          BIN=Stereo/stereo_kitti;                    D=Stereo ;;
  tum_mono)              BIN=Monocular/mono_tum;                     D=Monocular ;;
  tum_rgbd)              BIN=RGB-D/rgbd_tum;                         D=RGB-D ;;
  *) echo "unknown combination: ${DATASET} ${CONFIG}" >&2; usage ;;
esac

case "$DATASET" in
  euroc)
    [ -n "$SEQ" ] || { echo "euroc needs a sequence, e.g. MH01" >&2; exit 2; }
    SLAM_ARGS="$V $E/$D/EuRoC.yaml /datasets/EuRoC/$SEQ $E/$D/EuRoC_TimeStamps/$SEQ.txt live"
    TAG="euroc_${SEQ}_${CONFIG}" ;;
  kitti)
    [ -n "$SEQ" ] || { echo "kitti needs a sequence, e.g. 04" >&2; exit 2; }
    case "$SEQ" in
      00|01|02) YAML=KITTI00-02.yaml ;;
      03)       YAML=KITTI03.yaml ;;
      *)        YAML=KITTI04-12.yaml ;;
    esac
    SLAM_ARGS="$V $E/$D/$YAML /datasets/kitti_dataset/data_odometry_gray/dataset/sequences/$SEQ"
    TAG="kitti_${SEQ}_${CONFIG}" ;;
  tum)
    T=/datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk
    if [ "$CONFIG" = rgbd ]; then
      SLAM_ARGS="$V $E/$D/TUM1.yaml $T $E/RGB-D/associations/fr1_desk.txt"
    else
      SLAM_ARGS="$V $E/$D/TUM1.yaml $T"
    fi
    TAG="tum_fr1desk_${CONFIG}" ;;
esac

NAME=orbslam3r-monitor
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "results/live/$TAG"

echo "== ${DATASET} ${CONFIG} ${SEQ}  ->  results/live/${TAG}"
docker run -d --rm --name "$NAME" \
  --platform linux/arm64 \
  --shm-size=2g \
  -p "127.0.0.1:${PORT}:5900" \
  -e ORBSLAM3R_VIEWER=1 \
  -e XVFB_RESOLUTION=1600x900x24 \
  -e "VNC_PASSWORD=${VNC_PASSWORD}" \
  -v "${ROOT}:/workspace" \
  -v "$(cd "$ROOT/.." && pwd)/Datasets:/datasets:ro" \
  -w /workspace \
  orbslam3r/dev:24.04 \
  bash -c "
    # The entrypoint has already started Xvfb on \$DISPLAY.
    # -rfbauth, not -nopw: with no password x11vnc offers only RFB security
    # type 1 (None), and macOS Screen Sharing never completes that handshake.
    x11vnc -storepasswd \"\$VNC_PASSWORD\" /tmp/vncpass >/dev/null 2>&1
    x11vnc -display \$DISPLAY -forever -shared -rfbauth /tmp/vncpass \
           -rfbport 5900 -quiet -bg -o /workspace/results/live/${TAG}/x11vnc.log
    exec /workspace/build/bin/${BIN} ${SLAM_ARGS} \
      > /workspace/results/live/${TAG}/run.log 2>&1
  " >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# Wait for x11vnc to accept connections rather than guessing at a sleep.
for _ in $(seq 1 60); do
  if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then break; fi
  docker ps -q -f "name=$NAME" | grep -q . || { echo "container exited early:"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 0.5
done
nc -z 127.0.0.1 "$PORT" 2>/dev/null || { echo "VNC port ${PORT} never opened" >&2; exit 1; }

echo "== VNC ready at vnc://localhost:${PORT}"
echo "== password: ${VNC_PASSWORD}"
if [ "$OPEN" = 1 ]; then
  open "vnc://localhost:${PORT}"
  echo "== opened in Screen Sharing"
else
  echo "== open it with:  open vnc://localhost:${PORT}"
fi
echo "== Ctrl-C stops the run"
echo

docker logs -f "$NAME" 2>&1 || true
tail -n 20 "results/live/${TAG}/run.log" 2>/dev/null || true
