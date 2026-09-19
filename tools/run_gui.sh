#!/usr/bin/env bash
# Runs a whole dataset with the viewer on, one sequence after another, in a
# single window you can watch from start to finish.
#
#   ./tools/run_gui.sh euroc                     every EuRoC sequence, all 4 configurations
#   ./tools/run_gui.sh euroc stereo              one configuration
#   ./tools/run_gui.sh euroc stereo MH01 V203    named sequences only
#   ./tools/run_gui.sh kitti stereo
#   ./tools/run_gui.sh tum
#   ./tools/run_gui.sh all                       every dataset
#
#   --list       print the plan and exit, without running anything
#   --hold-each  hold every run's final map on screen, not just the last one's
#   --no-hold    close the window with the last frame, as upstream does
#   --vnc        export the screen over VNC instead of opening a desktop window
#   --no-open    (vnc) do not launch Screen Sharing, just print the URL
#   --port N     (vnc) local port, default 5900
#
# Configurations: euroc mono|stereo|mono_inertial|stereo_inertial,
#                 kitti mono|stereo, tum mono|rgbd. `all` is accepted everywhere.
#
# The difference from tools/monitor.sh is the plural: monitor.sh watches one
# sequence, this walks a dataset. The nested X server starts once and stays up
# for the whole series, so the window does not flicker in and out between runs.
#
# Trajectories land in results/gui/<tag>/, one directory per run, which is what
# lets tools/evaluate_ate.py score them afterwards.
#
# When the last run ends the window stays up showing the final map; Esc or the
# Stop button in the window ends it. Upstream closed the window with the last
# frame, which on KITTI 04 -- 27 seconds of driving -- looks like a crash.
#
# These are slow. Rendering is software llvmpipe inside the container, so a full
# EuRoC sweep is 44 runs and takes hours; --list first, and name the sequences
# you actually want to watch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/dataset_plan.sh

MODE=x11
PORT=5900
OPEN=1
LIST=0
HOLD=last
VNC_PASSWORD="${ORBSLAM3R_VNC_PASSWORD:-orbslam3r}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list)    LIST=1; shift ;;
    --x11)     MODE=x11; shift ;;
    --vnc)     MODE=vnc; shift ;;
    --no-open) OPEN=0; shift ;;
    --hold-each) HOLD=each; shift ;;
    --no-hold) HOLD=0; shift ;;
    --port)    PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
case "$DATASET" in
  euroc|kitti|tum|all) ;;
  *) echo "unknown dataset: $DATASET  (euroc|kitti|tum|all)" >&2; exit 2 ;;
esac

OUT=results/gui
PLAN=$OUT/plan.txt
mkdir -p "$OUT"
dataset_plan "$@" > "$PLAN"

COUNT=$(grep -c . "$PLAN" || true)
if [ "$COUNT" = 0 ]; then
  echo "nothing matched: $*" >&2
  exit 2
fi

echo "== ${COUNT} run(s) -> ${OUT}/<tag>  [${MODE}]"
cut -d'|' -f1 "$PLAN" | sed 's/^/   /'
if [ "$LIST" = 1 ]; then exit 0; fi
echo

NAME=orbslam3r-gui
docker rm -f "$NAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

COMMON=(
  --rm --name "$NAME"
  --platform linux/arm64
  --shm-size=2g
  -e ORBSLAM3R_VIEWER=1
  -e "ORBSLAM3R_VIEWER_HOLD=${HOLD}"
  -v "${ROOT}:/workspace"
  -v "$(cd "$ROOT/.." && pwd)/Datasets:/datasets:ro"
  -w /workspace
)

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

  echo "== a window titled 'ORB-SLAM3 Viewer' will open and stay up for the series"
  case "$HOLD" in
    last) echo "== when the last run ends its final map stays on screen: Esc or Stop in the window ends it" ;;
    each) echo "== each run holds its final map on screen: Esc or Stop in the window goes on to the next" ;;
  esac
  echo "== Ctrl-C stops everything"
  echo
  # DISPLAY is the OUTER server, the one Xephyr paints onto; the launcher moves
  # the app to the nested one. ORBSLAM3R_DISPLAY_MODE=x11 matters too: without it
  # the entrypoint takes its headless branch and starts a second, unused Xvfb.
  docker run "${COMMON[@]}" \
    -e DISPLAY=host.docker.internal:0 \
    -e ORBSLAM3R_DISPLAY_MODE=x11 \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    orbslam3r/dev:24.04 \
    /workspace/tools/container_x11_launch.sh --plan "/workspace/$PLAN" "/workspace/$OUT" \
    2>&1 | grep -v "Failed to attach to x11 shm" || true
  exit 0
fi

# --- VNC --------------------------------------------------------------------
# -rfbauth, not -nopw: with no password x11vnc offers only RFB security type 1
# (None), and macOS Screen Sharing never completes that handshake -- it sits on
# "Connecting..." forever. The port is published on 127.0.0.1 only either way.
docker run -d "${COMMON[@]}" \
  -p "127.0.0.1:${PORT}:5900" \
  -e XVFB_RESOLUTION=1600x900x24 \
  -e "VNC_PASSWORD=${VNC_PASSWORD}" \
  orbslam3r/dev:24.04 \
  bash -c "
    x11vnc -storepasswd \"\$VNC_PASSWORD\" /tmp/vncpass >/dev/null 2>&1
    x11vnc -display \$DISPLAY -forever -shared -rfbauth /tmp/vncpass \
           -rfbport 5900 -quiet -bg -o /workspace/${OUT}/x11vnc.log
    exec /workspace/tools/container_run_plan.sh /workspace/$PLAN /workspace/$OUT
  " >/dev/null

for _ in $(seq 1 60); do
  nc -z 127.0.0.1 "$PORT" 2>/dev/null && break
  docker ps -q -f "name=$NAME" | grep -q . || { echo "container exited early:"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 0.5
done
nc -z 127.0.0.1 "$PORT" 2>/dev/null || { echo "VNC port ${PORT} never opened" >&2; exit 1; }

echo "== VNC ready at vnc://localhost:${PORT}   password: ${VNC_PASSWORD}"
[ "$OPEN" = 1 ] && open "vnc://localhost:${PORT}" || echo "== open it with:  open vnc://localhost:${PORT}"
echo "== Ctrl-C stops the series"
echo
docker logs -f "$NAME" 2>&1 || true
