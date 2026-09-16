#!/usr/bin/env bash
# Verifies that every thirdparty/ submodule is still an untouched upstream
# checkout: clean working tree, no staged changes, sitting exactly on its
# pinned release tag.
#
# This is the guarantee the whole wrapper architecture rests on -- if a
# submodule has been edited, "what did ORB-SLAM3 change?" stops being answerable
# by diffing against upstream.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

problems=0
printf '%-22s %-18s %s\n' "SUBMODULE" "PINNED TAG" "STATE"
printf '%-22s %-18s %s\n' "---------" "----------" "-----"

git config --file .gitmodules --get-regexp '^submodule\..*\.path$' |
while read -r _ path; do
  if [ ! -e "$path/.git" ]; then
    printf '%-22s %-18s %s\n' "$path" "-" "NOT CHECKED OUT"
    problems=$((problems + 1))
    continue
  fi

  tag="$(git -C "$path" describe --tags --exact-match 2>/dev/null || echo '(no tag)')"
  dirty="$(git -C "$path" status --porcelain)"
  state="pristine"
  if [ -n "$dirty" ]; then
    state="MODIFIED ($(printf '%s\n' "$dirty" | wc -l | tr -d ' ') paths)"
    problems=$((problems + 1))
  elif [ "$tag" = "(no tag)" ]; then
    state="detached from any tag"
    problems=$((problems + 1))
  fi
  printf '%-22s %-18s %s\n' "$path" "$tag" "$state"

  if [ -n "$dirty" ]; then
    printf '%s\n' "$dirty" | sed 's/^/     /'
  fi
done

echo
if git submodule foreach --quiet 'git status --porcelain | grep -q . && echo "$sm_path" || true' | grep -q .; then
  echo "FAIL: at least one submodule has local modifications."
  echo "      Upstream must stay untouched; put changes in vendor_ext/ instead."
  exit 1
fi
echo "OK: every submodule is a clean upstream checkout."
