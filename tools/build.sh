#!/usr/bin/env bash
# Configures and builds inside the dev container.
#
#   ./tools/build.sh            incremental
#   ./tools/build.sh --clean    wipe build/ first
#
# Logs land in build/cmake.log and build/compile.log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLEAN=0
[ "${1:-}" = "--clean" ] && CLEAN=1
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

# The build directory is created on the host rather than inside the container:
# it sits on a bind mount, and removing then immediately rewriting it from the
# container side races the mount and fails with "No such file or directory".
[ "$CLEAN" = "1" ] && rm -rf build
mkdir -p build

./docker/run.sh -- bash -c "
  set -e
  cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    > /workspace/build/cmake.log 2>&1 || { tail -20 /workspace/build/cmake.log; exit 1; }
  cmake --build /workspace/build -j ${JOBS} > /workspace/build/compile.log 2>&1 || true
  errors=\$(grep -c 'error:' /workspace/build/compile.log || true)
  warnings=\$(grep -c 'warning:' /workspace/build/compile.log || true)
  echo \"errors=\${errors} warnings=\${warnings}\"
  if [ \"\${errors}\" != '0' ]; then
    grep -E 'error:' /workspace/build/compile.log | head -20
    exit 1
  fi
  echo '--- artefacts ---'
  ls /workspace/build/lib/ 2>/dev/null | head -3
  find /workspace/build/bin -type f 2>/dev/null | wc -l | xargs -I{} echo 'example binaries: {}'
  cd /workspace/build && ctest --output-on-failure 2>&1 | tail -4
"
