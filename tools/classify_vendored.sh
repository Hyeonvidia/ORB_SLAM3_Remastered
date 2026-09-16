#!/usr/bin/env bash
# Classifies every file ORB-SLAM3 vendored from an upstream project as either
#   UNTOUCHED  - the exact blob exists somewhere in upstream history, so
#                ORB-SLAM3 only froze an older revision; nothing was edited.
#   EDITED     - no upstream blob matches, so ORB-SLAM3 really changed it.
#
# Blob identity is the test, which is why upstream reformatting churn does not
# get misreported as an ORB-SLAM3 modification.
#
#   ./tools/classify_vendored.sh [g2o|Sophus|DBoW2|all]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

classify() {
  name="$1"; vendored_dir="$2"; upstream_repo="$3"
  echo "############ ${name} ############"
  echo "vendored : ${vendored_dir}"
  echo "upstream : ${upstream_repo}"

  # Every blob the upstream repository has ever stored, across all of history.
  # (cat-file --batch-all-objects walks the object database directly, so no
  # commit needs to be checked out and nothing is missed.)
  blobs="$(mktemp)"
  git -C "$upstream_repo" cat-file --batch-all-objects \
        --batch-check='%(objectname) %(objecttype)' |
    awk '$2 == "blob" { print $1 }' | LC_ALL=C sort -u > "$blobs"
  echo "upstream distinct blobs in history: $(wc -l < "$blobs" | tr -d ' ')"
  echo

  untouched=0; edited=0; edited_list=""
  while IFS= read -r f; do
    rel="${f#"$vendored_dir"/}"
    h="$(git hash-object "$f")"
    if LC_ALL=C grep -qx "$h" "$blobs"; then
      untouched=$((untouched+1))
      printf '  %-46s UNTOUCHED\n' "$rel"
    else
      edited=$((edited+1)); edited_list="${edited_list}${rel}"$'\n'
      printf '  %-46s EDITED   <-- ORB-SLAM3 change\n' "$rel"
    fi
  done < <(find "$vendored_dir" -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cc' \) | sort)

  rm -f "$blobs"
  echo
  echo "  == ${name}: ${untouched} untouched, ${edited} edited"
  if [ -n "$edited_list" ]; then
    echo "  == files ORB-SLAM3 actually changed:"
    printf '%s' "$edited_list" | sed 's/^/       /'
  fi
  echo
}

target="${1:-all}"
for t in g2o Sophus DBoW2; do
  [ "$target" = "all" ] || [ "$target" = "$t" ] || continue
  case "$t" in
    g2o)    classify g2o    reference/ORB_SLAM3/Thirdparty/g2o/g2o       thirdparty/g2o ;;
    Sophus) classify Sophus reference/ORB_SLAM3/Thirdparty/Sophus/sophus thirdparty/Sophus ;;
    DBoW2)  classify DBoW2  reference/ORB_SLAM3/Thirdparty/DBoW2         thirdparty/DBoW2 ;;
  esac
done
