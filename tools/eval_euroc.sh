#!/usr/bin/env bash
# Scores every trajectory in results/ against the right ground truth.
#
# Which reference to use is not a detail: ORB-SLAM3 reports pure-visual
# trajectories in the LEFT CAMERA frame and visual-inertial ones in the IMU BODY
# frame.  Scoring an inertial run against the left-camera ground truth leaves a
# camera-to-IMU offset that rotates with the trajectory and so does not cancel
# under a single global alignment -- on MH01 that alone inflated ATE from 4.7 cm
# to 7.5 cm.
#
# Alignment differs by configuration:
#   monocular          Sim(3).  The trajectory is only determined up to scale,
#                      so the recovered scale is a free parameter, not an error.
#   everything else    SE(3).  Stereo and inertial configurations observe metric
#                      scale, so aligning with Sim(3) would absorb a genuine
#                      scale error into the fit and flatter the result.  On MH01
#                      mono-inertial that is the difference between reporting
#                      8.4 cm and 3.0 cm while hiding a 2.2% scale error.
#                      For those runs the Sim(3) scale is reported separately as
#                      a diagnostic.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEQ="${1:-MH01}"

score() {   # trajectory, ground truth, extra flags -> "rmse pairs scale"
  ./docker/run.sh -- python3 /workspace/tools/evaluate_ate.py "$1" "$2" $3 2>/dev/null \
    | awk '/ATE RMSE/ {r=$4} /matched pairs/ {p=$4} /^scale / {s=$3} END {print r, p, s}'
}

run() {
  config="$1"; gt="$2"; align="$3"
  [ -f "results/f_${SEQ}_${config}.txt" ] || { printf '%-18s %s\n' "$config" "(not run)"; return; }
  traj="/workspace/results/f_${SEQ}_${config}.txt"

  case "$align" in
    sim3)
      read -r rmse pairs scale <<< "$(score "$traj" "$gt" --scale)"
      note="scale $scale (free: monocular)"
      ;;
    se3)
      read -r rmse pairs _ <<< "$(score "$traj" "$gt" "")"
      note=""
      ;;
    se3+scale)
      read -r rmse pairs _ <<< "$(score "$traj" "$gt" "")"
      read -r _ _ scale     <<< "$(score "$traj" "$gt" --scale)"
      err=$(awk -v s="$scale" 'BEGIN { printf "%.2f", (s - 1) * 100 }')
      note="scale error ${err}%"
      ;;
  esac
  printf '%-18s %-10s %-8s %s\n' "$config" "$rmse" "$pairs" "$note"
}

CAM_GT="/workspace/evaluation/Ground_truth/EuRoC_left_cam/${SEQ}_GT.txt"
IMU_GT="/datasets/EuRoC/${SEQ}/mav0/state_groundtruth_estimate0/data.csv"

echo "EuRoC ${SEQ}"
printf '%-18s %-10s %-8s %s\n' "CONFIG" "ATE RMSE" "PAIRS" "NOTE"
printf '%-18s %-10s %-8s %s\n' "------" "--------" "-----" "----"
run mono            "$CAM_GT" sim3
run stereo          "$CAM_GT" se3
run mono_inertial   "$IMU_GT" se3+scale
run stereo_inertial "$IMU_GT" se3+scale
