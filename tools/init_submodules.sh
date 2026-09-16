#!/usr/bin/env bash
# Adds every upstream dependency as a pristine git submodule pinned to a release tag.
# Re-runnable: skips submodules that are already registered.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# path|url|pinned tag
DEPS=(
  "thirdparty/g2o|https://github.com/RainerKuemmerle/g2o.git|20241228_git"
  "thirdparty/Sophus|https://github.com/strasdat/Sophus.git|1.24.6"
  "thirdparty/DBoW2|https://github.com/dorian3d/DBoW2.git|v1.1-free"
  "thirdparty/DLib|https://github.com/dorian3d/DLib.git|v1.1-free"
  "thirdparty/Pangolin|https://github.com/stevenlovegrove/Pangolin.git|v0.9.6"
  "reference/ORB_SLAM3|https://github.com/UZ-SLAMLab/ORB_SLAM3.git|v1.0-release"
)

for entry in "${DEPS[@]}"; do
  IFS='|' read -r path url tag <<< "$entry"
  if git config --file .gitmodules --get-regexp "submodule\..*\.path" | grep -qx "submodule.${path}.path ${path}"; then
    echo "== ${path}: already registered, syncing to ${tag}"
  else
    echo "== ${path}: adding ${url} @ ${tag}"
    git submodule add -q "$url" "$path"
  fi
  git -C "$path" fetch -q --tags origin
  git -C "$path" checkout -q "refs/tags/${tag}"
  echo "   -> $(git -C "$path" describe --tags --always) ($(git -C "$path" rev-parse --short HEAD))"
done

echo
echo "All submodules pinned. Summary:"
git submodule status
