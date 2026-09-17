#!/usr/bin/env bash
# Runs inside the dev container for the --x11 viewer mode: start a nested X
# server, give the viewer input focus, then run the SLAM binary under it.
#
#   container_x11_launch.sh <log_dir> <binary_path> [slam args...]
#   container_x11_launch.sh --plan <plan_file> <results_root>
#
# The --plan form runs a whole series under ONE nested server: the window opens
# once and stays up while each sequence takes its turn, instead of flickering in
# and out per run. Plan lines are `tag|binary|args`, the format
# tools/dataset_plan.sh emits; each run gets results_root/<tag> as its working
# directory, which is not optional -- the KITTI and TUM examples hard-code their
# output filenames, so runs sharing a directory overwrite each other.
#
# This lives in its own file rather than inline in a `docker run bash -c "..."`
# because the quoting there has to survive two shells, and the focus loop
# silently stopped working when it did not.
#
# WHY A NESTED X SERVER
#   Mesa cannot present to XQuartz from here: it logs "Failed to attach to x11
#   shm" once per frame and the window stays blank, because MIT-SHM has no
#   meaning between this Linux VM and macOS and Mesa 25's software X11 path has
#   no fallback. Xephyr owns a framebuffer in this container, so Mesa presents
#   to it locally with shared memory working; Xephyr then repaints its own
#   window on the outer server with plain XPutImage. It probes SHM once at
#   startup (hostx_init_shm), prints "Xephyr unable to use SHM XImages", and
#   switches to that path for good -- the graceful fallback Mesa lacks.
set -euo pipefail

MODE=single
if [ "${1:-}" = "--plan" ]; then
  MODE=plan
  PLAN="${2:?usage: $0 --plan <plan_file> <results_root>}"
  ROOT_OUT="${3:?usage: $0 --plan <plan_file> <results_root>}"
  LOG_DIR="$ROOT_OUT"
else
  LOG_DIR="${1:?usage: $0 <log_dir> <binary> [args...]}"
  BIN="${2:?usage: $0 <log_dir> <binary> [args...]}"
  shift 2
fi

OUTER_DISPLAY="${DISPLAY:?DISPLAY must point at the outer X server}"
INNER_DISPLAY="${ORBSLAM3R_INNER_DISPLAY:-:100}"
SCREEN="${ORBSLAM3R_XEPHYR_SCREEN:-1600x900x24}"

mkdir -p "$LOG_DIR"

# -screen ...x24 pins the depth instead of letting it be inherited.
# -no-host-grab stops a stray Ctrl+Shift on the host from capturing the
# keyboard and mouse into the nested server.
# -sw-cursor makes Xephyr draw the pointer into its own framebuffer. Without it
# the cursor is left to the host layer, which does not survive this transport,
# so it vanishes the moment it enters the window and the GUI becomes unusable.
Xephyr "$INNER_DISPLAY" -screen "$SCREEN" \
    -title 'ORB-SLAM3 Viewer' -resizeable -no-host-grab -sw-cursor -nolisten tcp \
    > "$LOG_DIR/xephyr.log" 2>&1 &
XEPHYR_PID=$!

export DISPLAY="$INNER_DISPLAY"
for _ in $(seq 1 30); do
  xdpyinfo >/dev/null 2>&1 && break
  sleep 1
done
if ! xdpyinfo >/dev/null 2>&1; then
  echo "Xephyr did not start on $INNER_DISPLAY (outer was $OUTER_DISPLAY):" >&2
  cat "$LOG_DIR/xephyr.log" >&2
  exit 1
fi

# Nothing inside Xephyr assigns input focus -- there is no window manager -- so
# keyboard events would go nowhere. Mouse buttons still reach the window under
# the pointer, but the viewer also takes key presses. The loop runs for the
# whole session because a series closes and reopens the window once per run.
focus_loop() {
  while true; do
    win=$(xdotool search --name 'ORB-SLAM3: Viewer' 2>/dev/null | head -1)
    if [ -n "$win" ]; then
      xdotool windowfocus "$win" >/dev/null 2>&1 || true
      xdotool windowactivate "$win" >/dev/null 2>&1 || true
    fi
    sleep 2
  done
}
focus_loop &
FOCUS_PID=$!

cleanup() {
  kill "$FOCUS_PID" 2>/dev/null || true
  kill "$XEPHYR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "$MODE" = single ]; then
  "$BIN" "$@" > "$LOG_DIR/run.log" 2>&1
  exit $?
fi

# --- series -----------------------------------------------------------------
# Not exec: the EXIT trap above is what stops Xephyr and the focus loop.
/workspace/tools/container_run_plan.sh "$PLAN" "$ROOT_OUT"
