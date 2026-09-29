#!/usr/bin/env bash
# What a change is judged by, in minutes: time per frame, accuracy, what place
# recognition found and peak memory, for the three sensor configurations the
# system is for -- monocular, stereo, RGB-D -- against the figures that were
# last accepted on this machine.
#
#   ./tools/check.sh              quick: KITTI 07 mono and stereo (one loop),
#                                 TUM fr1_desk RGB-D, EuRoC V101 stereo; 3 min
#   ./tools/check.sh robust       loops and merges: KITTI 05 mono and stereo
#                                 (three loops), EuRoC V101+V102 in one
#                                 session, mono and stereo (one merge); 5 min
#   ./tools/check.sh accept [set] what the last run of the set measured is
#                                 what the next ones are compared with
#   REPEATS=3 ./tools/check.sh    medians of three; one run of a monocular
#                                 sequence says little about its accuracy
#
# The runs share the machine four at a time, which costs each of them time per
# frame; they do so every time, so the comparison holds and the figure itself
# is not what the same run takes alone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/dataset_plan.sh

SET="${1:-quick}"
if [ "$SET" = accept ]; then
  SET="${2:-quick}"
  cp "results/check/$SET/summary.tsv" "results/check/reference_$SET.tsv"
  echo "results/check/reference_$SET.tsv is now what '$SET' is compared with"
  exit 0
fi

# Two sequences in one session: the arguments of the first, with the second's
# folder and timestamps put before the name of the output.
together() {
  local a b
  a=$(dataset_plan euroc "$1" "$2"); b=$(dataset_plan euroc "$1" "$3")
  awk -F'|' -v b="$b" -v tag="euroc_$2+$3_$1" '{
    split(b, fb, "|"); n = split($3, x, " "); split(fb[3], y, " ")
    printf "%s|%s|%s %s %s %s %s %s %s\n", tag, $2, x[1], x[2], x[3], x[4], y[3], y[4], x[n] }' <<< "$a"
}

OUT="results/check/$SET"
rm -rf "$OUT"; mkdir -p "$OUT"
case "$SET" in
  quick)
    { dataset_plan kitti all 07; dataset_plan tum rgbd; dataset_plan euroc stereo V101; } > "$OUT/plan.txt" ;;
  robust)
    { dataset_plan kitti all 05; together stereo V101 V102; together mono V101 V102; } > "$OUT/plan.txt" ;;
  *) echo "usage: $0 [quick|robust|accept [set]]" >&2; exit 2 ;;
esac
[ -x build/bin/Stereo/stereo_kitti ] || { echo "no build: run ./tools/build.sh first" >&2; exit 1; }
git describe --always --dirty > "$OUT/commit.txt"

(caffeinate -i -w $$ > /dev/null 2>&1 &)
./docker/run.sh -- python3 /workspace/tools/check_run.py "/workspace/$OUT" "${JOBS:-4}" "${REPEATS:-1}" \
  "/workspace/results/check/reference_$SET.tsv"
