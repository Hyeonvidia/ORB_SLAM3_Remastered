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

WHICH="${1:-all}"
JOBS="${JOBS:-4}"
OUT=results/matrix
mkdir -p "$OUT"

# tag|binary|settings|args...   (paths are container-side)
plan() {
  local V=/workspace/Vocabulary/ORBvoc.txt
  local E=/workspace/Examples

  if [ "$WHICH" = all ] || [ "$WHICH" = euroc ]; then
    for seq in MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203; do
      echo "euroc_${seq}_mono|Monocular/mono_euroc|$V $E/Monocular/EuRoC.yaml /datasets/EuRoC/$seq $E/Monocular/EuRoC_TimeStamps/$seq.txt t"
      echo "euroc_${seq}_stereo|Stereo/stereo_euroc|$V $E/Stereo/EuRoC.yaml /datasets/EuRoC/$seq $E/Stereo/EuRoC_TimeStamps/$seq.txt t"
      echo "euroc_${seq}_mono_inertial|Monocular-Inertial/mono_inertial_euroc|$V $E/Monocular-Inertial/EuRoC.yaml /datasets/EuRoC/$seq $E/Monocular-Inertial/EuRoC_TimeStamps/$seq.txt t"
      echo "euroc_${seq}_stereo_inertial|Stereo-Inertial/stereo_inertial_euroc|$V $E/Stereo-Inertial/EuRoC.yaml /datasets/EuRoC/$seq $E/Stereo-Inertial/EuRoC_TimeStamps/$seq.txt t"
    done
  fi

  if [ "$WHICH" = all ] || [ "$WHICH" = kitti ]; then
    local K=/datasets/kitti_dataset/data_odometry_gray/dataset/sequences
    # Only 00-10 have ground-truth poses; 11-21 are the evaluation split.
    for seq in 00 01 02 03 04 05 06 07 08 09 10; do
      case "$seq" in
        00|01|02) yaml=KITTI00-02.yaml ;;
        03)       yaml=KITTI03.yaml ;;
        *)        yaml=KITTI04-12.yaml ;;
      esac
      echo "kitti_${seq}_mono|Monocular/mono_kitti|$V $E/Monocular/$yaml $K/$seq"
      echo "kitti_${seq}_stereo|Stereo/stereo_kitti|$V $E/Stereo/$yaml $K/$seq"
    done
  fi

  if [ "$WHICH" = all ] || [ "$WHICH" = tum ]; then
    local T=/datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk
    echo "tum_fr1desk_mono|Monocular/mono_tum|$V $E/Monocular/TUM1.yaml $T"
    echo "tum_fr1desk_rgbd|RGB-D/rgbd_tum|$V $E/RGB-D/TUM1.yaml $T $E/RGB-D/associations/fr1_desk.txt"
  fi
}

plan > "$OUT/plan.full.txt"
touch "$OUT/status.txt"
# Drop anything already completed so an interrupted run resumes.
awk -F' ' '{print $1}' "$OUT/status.txt" | sort -u > "$OUT/.done"
awk -F'|' 'NR==FNR {done[$1]; next} !($1 in done)' "$OUT/.done" "$OUT/plan.full.txt" > "$OUT/plan.txt"
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
