#!/usr/bin/env bash
# Runs the v1.0 baseline and the remaster on KITTI 00-10, mono and stereo,
# headless, and scores both the same way, so the two can be compared.
#
#   ./tools/baseline/build.sh                  # once
#   ./tools/ab_kitti.sh                        # 110 runs, about two hours
#   ./tools/ab_kitti.sh 04 07                  # named sequences only
#   MONO_REPEATS=1 STEREO_REPEATS=1 ./tools/ab_kitti.sh 04
#
# Results land in results/ab/<baseline|remaster>_r<n>/<tag>/, and the summary
# in results/ab/summary.txt; ./docker/run.sh -- python3
# /workspace/tools/ab_summarise.py prints it again.
#
# Mono runs three times and stereo twice by default, because mono's
# run-to-run spread is the larger (README, "Status"). The two builds are
# interleaved in one queue, four at a time, so that whatever load the machine
# is under falls on both alike -- a sequential run of one build against a
# parallel run of the other is a different measurement. Resumable: a run
# whose directory holds a "done" marker is skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/dataset_plan.sh

MONO_REPEATS="${MONO_REPEATS:-3}"
STEREO_REPEATS="${STEREO_REPEATS:-2}"
JOBS="${JOBS:-4}"
BASE=results/baseline/ORB_SLAM3
[ -x "$BASE/Examples/Monocular/mono_kitti" ] || { echo "no baseline build: run ./tools/baseline/build.sh first" >&2; exit 1; }
[ -x build/bin/Monocular/mono_kitti ] || { echo "no remaster build: run ./tools/build.sh first" >&2; exit 1; }

mkdir -p results/ab
dataset_plan kitti all "$@" | grep -E '^kitti_[0-9]+_(mono|stereo)\|' | sort -u > results/ab/plan.txt
: > results/ab/jobs.txt
while IFS='|' read -r tag binrel args; do
  case "$tag" in *_mono) reps=$MONO_REPEATS ;; *) reps=$STEREO_REPEATS ;; esac
  kind=${binrel%/*}; name=${binrel#*/}
  for r in $(seq 1 "$reps"); do
    echo "baseline_r$r|$tag|/workspace/$BASE/Examples/$kind/$name|$args" >> results/ab/jobs.txt
    echo "remaster_r$r|$tag|/workspace/build/bin/$kind/$name|$args" >> results/ab/jobs.txt
  done
done < results/ab/plan.txt
echo "== $(grep -c . results/ab/jobs.txt) runs, ${JOBS} at a time -> results/ab/"

./docker/run.sh -- bash -c '
run_one() {
  set=$1; tag=$2; bin=$3; shift 3
  d=/workspace/results/ab/$set/$tag; mkdir -p "$d"; cd "$d"
  [ -f done ] && exit 0
  case "$set" in
    baseline*) O=/workspace/results/baseline/ORB_SLAM3
               export LD_LIBRARY_PATH=$O/Thirdparty/DBoW2/lib:$O/Thirdparty/g2o/lib:$O/lib:${LD_LIBRARY_PATH:-} ;;
  esac
  s=$(date +%s)
  ORBSLAM3R_VIEWER=0 "$bin" "$@" > run.log 2>&1; rc=$?
  echo "$set $tag rc=$rc secs=$(( $(date +%s) - s ))" | tee -a /workspace/results/ab/status.txt
  [ "$rc" = 0 ] && touch done
}
export -f run_one
while IFS="|" read -r set tag bin rest; do printf "%s\0%s\0%s\0%s\0" "$set" "$tag" "$bin" "$rest"; done < /workspace/results/ab/jobs.txt |
  xargs -0 -n 4 -P '"$JOBS"' bash -c '"'"'run_one "$0" "$1" "$2" $3'"'"'
python3 /workspace/tools/ab_summarise.py | tee /workspace/results/ab/summary.txt'
