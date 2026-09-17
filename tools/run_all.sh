#!/usr/bin/env bash
# Runs the whole dataset matrix -- EuRoC, KITTI and TUM -- inside one container,
# N sequences at a time, each in its own working directory.
#
# Per-run directories are not optional: the KITTI and TUM examples hard-code
# their output filenames ("CameraTrajectory.txt", "KeyFrameTrajectory.txt"), so
# concurrent runs sharing a cwd would overwrite each other's trajectories.
#
#   ./tools/run_all.sh            all three datasets
#   ./tools/run_all.sh euroc      one dataset (euroc|kitti|tum)
#   JOBS=6 ./tools/run_all.sh
#
# Re-runnable: any tag already recorded in results/matrix/status.txt is skipped,
# so an interrupted matrix resumes instead of starting over. Delete status.txt
# (or the tag's line) to force a rerun.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/dataset_plan.sh

WHICH="${1:-all}"
JOBS="${JOBS:-4}"
OUT=results/matrix
mkdir -p "$OUT"


dataset_plan "$WHICH" > "$OUT/plan.full.txt"
touch "$OUT/status.txt"
# Drop anything already completed so an interrupted run resumes.
awk -F' ' '{print $1}' "$OUT/status.txt" | sort -u > "$OUT/.done"
# Read the done-list in BEGIN rather than with the usual NR==FNR two-file
# trick: awk never enters the body for an empty first file, so NR==FNR stayed
# true while reading the plan and every run looked already-done. A fresh matrix
# -- the case where status.txt is empty -- therefore ran nothing and exited 0.
awk -F'|' -v donefile="$OUT/.done" \
    'BEGIN { while ((getline line < donefile) > 0) done[line] } !($1 in done)' \
    "$OUT/plan.full.txt" > "$OUT/plan.txt"
TOTAL=$(wc -l < "$OUT/plan.txt" | tr -d ' ')
DONE=$(wc -l < "$OUT/.done" | tr -d ' ')
echo "== ${TOTAL} runs left (${DONE} already done), ${JOBS} at a time -> results/matrix/"
[ "$TOTAL" -eq 0 ] && { echo "nothing to do"; exit 0; }

./docker/run.sh -- bash -c "
set -u
run_one() {
  tag=\$1; bin=\$2; shift 2
  d=/workspace/results/matrix/\$tag
  mkdir -p \"\$d\"
  cd \"\$d\"
  start=\$(date +%s)
  if /workspace/build/bin/\$bin \"\$@\" > run.log 2>&1; then rc=0; else rc=\$?; fi
  echo \"\$tag rc=\$rc secs=\$(( \$(date +%s) - start ))\" >> /workspace/results/matrix/status.txt
}
export -f run_one
while IFS='|' read -r tag bin rest; do
  printf '%s\0%s\0%s\0' \"\$tag\" \"\$bin\" \"\$rest\"
done < /workspace/results/matrix/plan.txt |
xargs -0 -n 3 -P ${JOBS} bash -c 'run_one \"\$0\" \"\$1\" \$2'
echo '== matrix complete'
sort /workspace/results/matrix/status.txt | tail -5
"
