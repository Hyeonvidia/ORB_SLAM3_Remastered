#!/usr/bin/env bash
# Container entrypoint: prepares a display for Pangolin, then runs the command.
set -euo pipefail

mode="${ORBSLAM3R_DISPLAY_MODE:-headless}"

case "$mode" in
  headless)
    # A virtual X server means Pangolin-linked binaries start even with no
    # display attached; the viewer itself is normally turned off in the config.
    if [ -z "${DISPLAY:-}" ]; then
      export DISPLAY=:99
      if ! xdpyinfo -display :99 >/dev/null 2>&1; then
        Xvfb :99 -screen 0 "${XVFB_RESOLUTION:-1600x1200x24}" -nolisten tcp >/tmp/xvfb.log 2>&1 &
        for _ in $(seq 1 50); do
          xdpyinfo -display :99 >/dev/null 2>&1 && break
          sleep 0.1
        done
      fi
    fi
    # llvmpipe: software GL, so OpenGL calls succeed without a host GPU.
    export LIBGL_ALWAYS_SOFTWARE=1
    # The Pangolin viewer is technically usable here, but software rendering
    # makes a dataset run unusably slow, so batch runs default to no viewer.
    # Override per-command with ORBSLAM3R_VIEWER=1.
    export ORBSLAM3R_VIEWER="${ORBSLAM3R_VIEWER:-0}"
    # Silences the "XDG_RUNTIME_DIR is invalid or not set" warning that GLFW /
    # Wayland client code emits inside a container.
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
    mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
    # Silences the "XDG_RUNTIME_DIR is invalid or not set" warning that GLFW /
    # Wayland client code emits inside a container.
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
    mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
    ;;
  x11)
    if [ -z "${DISPLAY:-}" ]; then
      echo "[entrypoint] ORBSLAM3R_DISPLAY_MODE=x11 but DISPLAY is unset." >&2
      echo "[entrypoint] On macOS: start XQuartz, run 'xhost +localhost'," >&2
      echo "[entrypoint] then pass -e DISPLAY=host.docker.internal:0." >&2
      exit 1
    fi
    ;;
  *)
    echo "[entrypoint] unknown ORBSLAM3R_DISPLAY_MODE='$mode' (expected headless|x11)" >&2
    exit 1
    ;;
esac

if [ -t 1 ] && [ "${ORBSLAM3R_QUIET:-0}" != "1" ]; then
  echo "== ORB_SLAM3_Remastered dev container"
  echo "   display mode : ${mode} (DISPLAY=${DISPLAY:-unset})"
  echo "   dependencies : /opt/orbslam3r  (cat THIRDPARTY_MANIFEST.txt for pins)"
  echo "   workspace    : $(pwd)"
  echo
fi

exec "$@"
