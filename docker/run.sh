#!/usr/bin/env bash
# Starts a dev container.
#   ./docker/run.sh                 # headless shell
#   ./docker/run.sh --gui           # viewer over XQuartz
#   ./docker/run.sh -- ./build/foo  # run one command and exit
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS="${DATASETS:-$(cd "$ROOT/.." && pwd)/Datasets}"

MODE=headless
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gui) MODE=x11; shift ;;
    --)    shift; ARGS=("$@"); break ;;
    *)     ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} -eq 0 ] && ARGS=(/bin/bash)

ENVS=(-e "ORBSLAM3R_DISPLAY_MODE=${MODE}")
if [ "$MODE" = "x11" ]; then
  if ! pgrep -qx Xquartz 2>/dev/null; then
    echo "[run.sh] XQuartz does not look like it is running."
    echo "[run.sh] Start it, enable 'Allow connections from network clients',"
    echo "[run.sh] then run: xhost +localhost"
  fi
  ENVS+=(-e "DISPLAY=host.docker.internal:0" -e "LIBGL_ALWAYS_INDIRECT=1")
fi

MOUNTS=(-v "${ROOT}:/workspace" -v "orbslam3r_ccache:/root/.ccache")
if [ -d "$DATASETS" ]; then
  MOUNTS+=(-v "${DATASETS}:/datasets:ro")
else
  echo "[run.sh] note: dataset directory '${DATASETS}' not found; /datasets not mounted."
fi

# Allocate a TTY only when there actually is one, so the script also works
# from CI, pipes and non-interactive tooling.
TTY_FLAGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS+=(-t); fi

exec docker run --rm "${TTY_FLAGS[@]}" \
  --platform linux/arm64 \
  --shm-size=2g \
  -w /workspace \
  "${ENVS[@]}" "${MOUNTS[@]}" \
  orbslam3r/dev:24.04 "${ARGS[@]}"
