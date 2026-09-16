#!/usr/bin/env bash
# Regenerates docker/thirdparty_manifest.txt from the current submodule state.
# Run after any submodule bump, before rebuilding the thirdparty image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT=docker/thirdparty_manifest.txt

{
  echo "ORB_SLAM3_Remastered — third-party dependency manifest"
  echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  printf '%-22s %-46s %-18s %s\n' "SUBMODULE" "UPSTREAM" "PINNED TAG" "COMMIT"
  printf '%-22s %-46s %-18s %s\n' "---------" "--------" "----------" "------"
  git config --file .gitmodules --get-regexp '^submodule\..*\.path$' |
  while read -r key path; do
    name="${key#submodule.}"; name="${name%.path}"
    url=$(git config --file .gitmodules --get "submodule.${name}.url")
    if [ -d "$path/.git" ] || [ -f "$path/.git" ]; then
      tag=$(git -C "$path" describe --tags --exact-match 2>/dev/null || echo "-")
      sha=$(git -C "$path" rev-parse --short HEAD)
    else
      tag="(not checked out)"; sha="-"
    fi
    printf '%-22s %-46s %-18s %s\n' "$path" "$url" "$tag" "$sha"
  done
  echo
  echo "All entries above are UNMODIFIED upstream checkouts."
  echo "ORB-SLAM3's changes to them live in vendor_ext/ — see docs/WRAPPERS.md."
} > "$OUT"

cat "$OUT"
