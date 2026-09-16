#!/usr/bin/env bash
# Builds the three image layers in order: base -> thirdparty -> dev.
#
#   ./docker/build_images.sh              # build all three
#   ./docker/build_images.sh thirdparty   # build base (cached) + thirdparty
#   ./docker/build_images.sh --no-cache   # force a full rebuild
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PLATFORM="${PLATFORM:-linux/arm64}"
TAG="${TAG:-24.04}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
# bash 3.2 (the macOS system shell) treats "${arr[@]}" on an empty array as an
# unbound variable under `set -u`, hence the ${arr[@]+...} guards below.
EXTRA=()
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --no-cache|--pull|--progress=*) EXTRA+=("$arg") ;;
    base|thirdparty|dev)            TARGETS+=("$arg") ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=(base thirdparty dev); fi

want() { printf '%s\n' "${TARGETS[@]}" | grep -qx "$1"; }

# The thirdparty image embeds a manifest of the pinned submodule revisions;
# regenerate it so the image can never disagree with the repository.
if want thirdparty; then
  echo "== regenerating docker/thirdparty_manifest.txt"
  ./tools/gen_manifest.sh >/dev/null
fi

build() {
  local name="$1"; shift
  echo
  echo "=============================================================="
  echo "  building orbslam3r/${name}:${TAG}  (${PLATFORM})"
  echo "=============================================================="
  docker build \
    --platform "${PLATFORM}" \
    -t "orbslam3r/${name}:${TAG}" \
    -f "docker/Dockerfile.${name}" \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@" .
}

# if/fi rather than `want X && build X`: with `set -e` a false `want` would
# abort the script, and `|| true` would swallow real build failures.
if want base; then
  build base
fi
if want thirdparty; then
  build thirdparty \
    --build-arg "BASE_IMAGE=orbslam3r/base:${TAG}" \
    --build-arg "JOBS=${JOBS}"
fi
if want dev; then
  build dev \
    --build-arg "THIRDPARTY_IMAGE=orbslam3r/thirdparty:${TAG}"
fi

echo
echo "== done"
docker images --filter "reference=orbslam3r/*" \
  --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
