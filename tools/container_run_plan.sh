#!/usr/bin/env bash
# Runs a plan, one sequence at a time, inside the container.
#
#   container_run_plan.sh <plan_file> <results_root>
#
# Plan lines are `tag|binary|args`, the format tools/dataset_plan.sh emits.
# Each run gets results_root/<tag> as its working directory. That is not a
# convenience: the KITTI and TUM examples hard-code their output filenames
# ("CameraTrajectory.txt", "KeyFrameTrajectory.txt"), so runs sharing a
# directory would overwrite each other's trajectories.
#
# Sequential on purpose. The point of these runs is watching them, and the
# viewer only has one window.
#
# ORBSLAM3R_VIEWER_HOLD says what happens to the window when a run ends:
#   last   the last run of the plan holds its final map on screen until Esc
#          or Stop is pressed in the window (the default under run_gui.sh)
#   each   every run does, so a series pauses after each sequence
#   0      nothing: the window closes with the last frame
# The binary itself only understands 1, so this is where last/each are turned
# into a per-run 1.
set -uo pipefail

PLAN="${1:?usage: $0 <plan_file> <results_root>}"
OUT_ROOT="${2:?usage: $0 <plan_file> <results_root>}"

total=$(grep -c . "$PLAN" || true)
n=0
fail=0
started=$(date +%s)

while IFS='|' read -r tag binary args; do
  [ -n "${tag:-}" ] || continue
  n=$((n + 1))
  out="$OUT_ROOT/$tag"
  mkdir -p "$out"
  hold=0
  case "${ORBSLAM3R_VIEWER_HOLD:-0}" in
    each|1) hold=1 ;;
    last)   [ "$n" = "$total" ] && hold=1 ;;
  esac
  printf '[%2d/%2d] %-32s ' "$n" "$total" "$tag"
  t0=$(date +%s)
  if (cd "$out" && ORBSLAM3R_VIEWER_HOLD=$hold "/workspace/build/bin/$binary" $args > run.log 2>&1); then
    printf 'ok    %4ds\n' "$(( $(date +%s) - t0 ))"
  else
    printf 'FAIL  %4ds   see %s/run.log\n' "$(( $(date +%s) - t0 ))" "${out#/workspace/}"
    fail=$((fail + 1))
  fi
done < "$PLAN"

echo "== ${n} run(s), ${fail} failed, $(( ($(date +%s) - started) / 60 )) min total"
[ "$fail" = 0 ]
