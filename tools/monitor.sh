#!/usr/bin/env bash
# Watch the ORB-SLAM3 viewer live from macOS while a sequence runs.
#
#   ./tools/monitor.sh euroc stereo MH01
#   ./tools/monitor.sh euroc stereo-inertial V203
#   ./tools/monitor.sh kitti stereo 04
#   ./tools/monitor.sh tum rgbd
#
# TWO WAYS TO SEE IT
#   --x11   (default)  a real window on the macOS desktop, through a nested X
#                      server. Nothing to connect to, no password, and the mouse
#                      works. Needs XQuartz, which this script starts and
#                      authorises for you.
#   --vnc              the container renders to its own Xvfb and x11vnc exports
#                      the screen; macOS opens it with Screen Sharing. Use it
#                      when XQuartz is not installed, or from a remote machine.
#
# WHY --x11 GOES THROUGH XEPHYR RATHER THAN STRAIGHT TO XQUARTZ
#   Pointing the viewer at XQuartz directly gets you a window that appears with
#   the right title and size and then stays blank. Rendering is not the problem
#   -- the probe in tools/glprobe reports a correct viewport, frames drawn and
#   no GL error -- presentation is. Mesa logs
#
#       MESA: error: Failed to attach to x11 shm
#
#   exactly once per frame, and the window never fills in. MIT-SHM cannot work
#   between the container's Linux VM and macOS, since there is no shared memory
#   segment across two kernels, and Mesa 25's software X11 path has no working
#   fallback. LIBGL_KOPPER_DISABLE, LIBGL_DRI3_DISABLE, GALLIUM_DRIVER=softpipe
#   and LIBGL_ALWAYS_INDIRECT all leave it unchanged.
#
#   Xephyr breaks the chain in the right place. It is an X server that runs in
#   the container and owns a framebuffer there, so Mesa presents to it locally
#   with shared memory working normally. Xephyr then repaints its own window on
#   XQuartz -- and unlike Mesa it handles the missing extension, logging
#   "Xephyr unable to use SHM XImages" once and falling back to plain XPutImage.
#
#   --no-open   (vnc) do not launch Screen Sharing; just print the URL
#   --port N    (vnc) use a different local port (default 5900)
#
# The VNC password defaults to "orbslam3r"; override with ORBSLAM3R_VNC_PASSWORD.
# It is not optional: macOS Screen Sharing never finishes the handshake against
# a server that offers only RFB security type 1 (None) -- it sits on
# "Connecting..." forever. A password makes x11vnc offer type 2, VNC
# Authentication, which Apple's client does support. The port is published on
# 127.0.0.1 only either way.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT=5900
OPEN=1
MODE=x11
VNC_PASSWORD="${ORBSLAM3R_VNC_PASSWORD:-orbslam3r}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --x11)     MODE=x11; shift ;;
    --vnc)     MODE=vnc; shift ;;
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

echo "== ${DATASET} ${CONFIG} ${SEQ}  ->  results/live/${TAG}  [${MODE}]"

COMMON=(
  --rm --name "$NAME"
  --platform linux/arm64
  --shm-size=2g
  -e ORBSLAM3R_VIEWER=1
  -v "${ROOT}:/workspace"
  -v "$(cd "$ROOT/.." && pwd)/Datasets:/datasets:ro"
  -w /workspace
)

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

if [ "$MODE" = x11 ]; then
  if ! pgrep -qx Xquartz 2>/dev/null; then
    echo "== starting XQuartz"
    open -a XQuartz
    for _ in $(seq 1 30); do pgrep -qx Xquartz 2>/dev/null && break; sleep 1; done
  fi
  pgrep -qx Xquartz 2>/dev/null || {
    echo "XQuartz is not available. Install it, or use Screen Sharing instead:" >&2
    echo "  $0 --vnc $*" >&2
    exit 1
  }
  # The container connects over TCP, so the X server has to allow it.
  /opt/X11/bin/xhost +localhost >/dev/null 2>&1 || true

  echo "== a window titled 'ORB-SLAM3 Viewer' will open on your desktop"
  echo "== Ctrl-C stops the run"
  echo
  # DISPLAY is the OUTER server, which is what Xephyr paints onto; the app is
  # switched to :100 below. ORBSLAM3R_DISPLAY_MODE=x11 matters too: without it
  # the entrypoint takes its headless branch and starts a second, unused Xvfb.
  # DISPLAY is the OUTER server, which is what Xephyr paints onto; the launcher
  # switches the app to the nested one. ORBSLAM3R_DISPLAY_MODE=x11 matters too:
  # without it the entrypoint takes its headless branch and starts a second,
  # unused Xvfb.
  docker run "${COMMON[@]}" \
    -e DISPLAY=host.docker.internal:0 \
    -e ORBSLAM3R_DISPLAY_MODE=x11 \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    orbslam3r/dev:24.04 \
    /workspace/tools/container_x11_launch.sh \
      "/workspace/results/live/${TAG}" \
      "/workspace/build/bin/${BIN}" ${SLAM_ARGS} \
    2>&1 | grep -v "Failed to attach to x11 shm" || true
  tail -n 20 "results/live/${TAG}/run.log" 2>/dev/null || true
  exit 0
fi

# --- VNC -------------------------------------------------------------------
docker run -d "${COMMON[@]}" \
  -p "127.0.0.1:${PORT}:5900" \
  -e XVFB_RESOLUTION=1600x900x24 \
  -e "VNC_PASSWORD=${VNC_PASSWORD}" \
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
