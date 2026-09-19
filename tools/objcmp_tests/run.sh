#!/usr/bin/env bash
# Checks tools/objcmp.py against small before/after pairs, compiled the way the
# project is (GCC, -O3 -fPIC): every case under must_change/ is a real change
# of behaviour and has to come out CHANGED; every case under must_pass/ moves
# or reorders code without changing what it does and must not.
#
# known_false_positive/ holds the cases objcmp's docstring lists as known
# limits: they change nothing, and objcmp says CHANGED anyway (statics reached
# through a section anchor). They are expected to come out CHANGED, so that
# the day one stops doing so the limit is known to be gone.
# known_false_negative/ is the other side of the same list: a real change
# objcmp does not see (two switch cases swapping bodies, which only the
# anonymous jump table records). They are expected to pass.
#
#   ./docker/run.sh -- /workspace/tools/objcmp_tests/run.sh
#
# A case is a directory with before/ and after/ holding .cpp files, plus any
# headers they include as "../name". A file named renames, one OLD=NEW per
# line, is passed to objcmp as --rename.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
failed=0

for kind in must_change must_pass known_false_positive known_false_negative; do
  for dir in "$HERE/$kind"/*/; do
    name="$(basename "$dir")"
    ok=1
    for side in before after; do
      mkdir -p "$OUT/$kind/$name/$side"
      for src in "$dir$side"/*.cpp; do
        g++ -O3 -DNDEBUG -std=gnu++17 -fPIC -c "$src" \
            -o "$OUT/$kind/$name/$side/$(basename "${src%.cpp}").o" || ok=0
      done
    done
    if [ "$ok" = 0 ]; then
      echo "FAIL  $kind/$name  (does not compile)"
      failed=1
      continue
    fi
    args=()
    if [ -f "$dir/renames" ]; then
      while IFS= read -r r; do args+=(--rename "$r"); done < "$dir/renames"
    fi
    python3 "$HERE/../objcmp.py" "${args[@]}" "$OUT/$kind/$name/before" "$OUT/$kind/$name/after" \
        > "$OUT/log" 2>&1
    rc=$?
    verdict="$(grep -E '^[a-zA-Z-]+ +[^ ]+\.o$' "$OUT/log" | awk '{print $1}' | sort -u | tr '\n' ' ')"
    case "$kind" in must_change|known_false_positive) want=1 ;; *) want=0 ;; esac
    if [ "$rc" = "$want" ]; then
      echo "ok    $kind/$name  ${verdict:-bit-identical}"
    else
      echo "FAIL  $kind/$name  exit $rc  ${verdict:-bit-identical}"
      sed 's/^/      /' "$OUT/log" | head -20
      failed=1
    fi
  done
done
exit "$failed"
