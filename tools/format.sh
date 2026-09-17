#!/usr/bin/env bash
# Applies .clang-format to the code this project owns.
#
#   ./tools/format.sh            rewrite the files in place
#   ./tools/format.sh --check    report what would change, touch nothing (exit 1 if any)
#   ./tools/format.sh --diff     show what --check counted
#
# Runs inside the dev container, never on the host: clang-format's output
# changes between major versions, so the formatting a file gets must not depend
# on which machine touched it. The container pins 18.1.3.
#
# WHAT IS NOT FORMATTED
#   thirdparty/  pinned upstream submodules, checked out byte-identical
#   reference/   pristine ORB-SLAM3 v1.0, the baseline every delta is measured
#                against -- reformatting it would silently rewrite the answer
# Neither is listed below, and both also carry their own .clang-format with
# DisableFormat, so an editor that formats on save cannot reach them either.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE=write
case "${1:-}" in
  --check) MODE=check ;;
  --diff)  MODE=diff ;;
  "")      ;;
  *) echo "usage: $0 [--check|--diff]" >&2; exit 2 ;;
esac

# The find runs inside the container so the file list is built where the files
# are formatted; -print0/-0 keeps it correct if a path ever gains a space.
# tools/smoke_test and tools/glprobe are in the list because the root
# .clang-format reaches them too; leaving them out would let an editor save
# reformat them while --check still called the tree clean.
FIND='find /workspace/src /workspace/include /workspace/Examples /workspace/vendor_ext \
        /workspace/tools/smoke_test /workspace/tools/glprobe \
        -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -print0'

case "$MODE" in
  write)
    ./docker/run.sh -- bash -c "$FIND | xargs -0 clang-format -i --style=file"
    echo "== formatted; net change:"
    git --no-pager diff --shortstat -- src include Examples vendor_ext
    ;;
  check|diff)
    # --dry-run -Werror reports differences on stderr and exits non-zero
    # without writing anything.
    OUT=$(./docker/run.sh -- bash -c "$FIND | xargs -0 clang-format --dry-run -Werror --style=file" 2>&1 || true)
    # Unique paths, not matching lines: clang-format reports one line per
    # violation, so counting matches called a single file with four misplaced
    # line breaks "4 files".
    N=$(printf '%s\n' "$OUT" | grep "code should be clang-formatted" | cut -d: -f1 | sort -u | grep -c . || true)
    if [ "$N" = "0" ]; then
      echo "clean: every source already matches .clang-format"
      exit 0
    fi
    [ "$MODE" = diff ] && printf '%s\n' "$OUT"
    [ "$N" = 1 ] && SUBJ="1 file differs" || SUBJ="${N} files differ"
    echo "${SUBJ} from .clang-format  (run ./tools/format.sh to fix)"
    exit 1
    ;;
esac
