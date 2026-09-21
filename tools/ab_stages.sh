#!/usr/bin/env bash
# Says WHICH STAGE of tracking and local mapping differs between two commits of
# the remaster: builds both with REGISTER_TIMES, runs them in pairs, one of each
# at a time, and prints every stage's mean side by side with the difference in
# each pair.
#
#   ./tools/ab_stages.sh e3e0ae5 a1b25ae           # six pairs on 04 stereo, 07 mono, 07 stereo
#   REPEATS=3 ./tools/ab_stages.sh HEAD~1 HEAD 04 07
#
# tools/ab_kitti.sh times whole runs, several at once, and that cannot see a
# percent: two builds of the same source, run through it side by side, came out
# 1.5 % apart. This can. A stage the change never touched (ORB extraction,
# stereo matching) is the control: if it moves as much as the stage in question,
# what is being looked at is the machine.
#
# Each commit is built once, in its own worktree under results/, and kept.
# Results land in results/ab_stages/<a>_vs_<b>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/dataset_plan.sh

[ $# -ge 2 ] || { sed -n '2,18p' "$0"; exit 1; }
A=$(git rev-parse --short "$1"); B=$(git rev-parse --short "$2"); shift 2
REPEATS="${REPEATS:-6}"
OUT=results/ab_stages/${A}_vs_${B}
mkdir -p "$OUT"

for c in "$A" "$B"; do
  WT=results/wt_$c
  [ -d "$WT" ] || git worktree add --detach "$WT" "$c" > /dev/null
  [ -x "$WT/build_times/bin/Stereo/stereo_kitti" ] && continue
  mkdir -p "$WT/build_times"
  echo "== building $c with REGISTER_TIMES"
  ./docker/run.sh -- bash -c "
    cmake -S /workspace/$WT -B /workspace/$WT/build_times -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS=-DREGISTER_TIMES -DORBSLAM3R_BUILD_SMOKE_TEST=OFF > /workspace/$WT/build_times/cmake.log 2>&1 &&
    cmake --build /workspace/$WT/build_times -j 12 > /workspace/$WT/build_times/compile.log 2>&1" ||
    { grep -m5 'error' "$WT/build_times/compile.log"; exit 1; }
done

# By default three short runs. 04 mono is left out: 271 frames, of which the
# monocular initialisation takes a varying share, and builds that differ in
# nothing tracking executes came out between -8 % and +13 % apart on it.
if [ $# -gt 0 ]; then dataset_plan kitti all "$@" | grep -E '^kitti_[0-9]+_(mono|stereo)\|'
else dataset_plan kitti all 04 07 | grep -E '^kitti_(04_stereo|07_mono|07_stereo)\|'; fi | sort -u > "$OUT/plan.txt"
: > "$OUT/jobs.txt"
for r in $(seq 1 "$REPEATS"); do
  # Which commit of a pair starts first alternates, so neither always does.
  if [ $((r % 2)) = 1 ]; then order="$A $B"; else order="$B $A"; fi
  while IFS='|' read -r tag binrel args; do
    for c in $order; do echo "${c}_r$r|$tag|/workspace/results/wt_$c/build_times/bin/$binrel|$args" >> "$OUT/jobs.txt"; done
  done < "$OUT/plan.txt"
done
echo "== $(grep -c . "$OUT/jobs.txt") runs, two at a time -> $OUT/"

./docker/run.sh -- bash -c '
OUT=/workspace/'"$OUT"'
run_one() {
  set=$1; tag=$2; bin=$3; shift 3
  d=$OUT/$set/$tag; mkdir -p "$d"; cd "$d"
  [ -f done ] && exit 0
  ORBSLAM3R_VIEWER=0 "$bin" "$@" > run.log 2>&1; rc=$?
  echo "$set $tag rc=$rc" >> $OUT/status.txt
  [ "$rc" = 0 ] && touch done
}
export -f run_one; export OUT
while IFS="|" read -r set tag bin rest; do printf "%s\0%s\0%s\0%s\0" "$set" "$tag" "$bin" "$rest"; done < $OUT/jobs.txt |
  xargs -0 -n 4 -P 2 bash -c '"'"'run_one "$0" "$1" "$2" $3'"'"''
python3 tools/ab_stages_summarise.py "$OUT" "$A" "$B" | tee "$OUT/summary.txt"
