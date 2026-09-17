#!/usr/bin/env bash
# The dataset table: which binary and which arguments each (dataset, sensor
# configuration, sequence) needs. Sourced, not executed.
#
#   dataset_plan <dataset> [config] [sequence ...]
#
# prints one line per run:
#
#   tag|binary|args...
#
# with container-side paths. `all` is accepted for both dataset and config, and
# an empty sequence list means every sequence that has ground truth.
#
# This lives on its own because run_all.sh, run_gui.sh and run_euroc.sh all need
# it, and three copies of the same table drift apart the first time a settings
# file is renamed.

ORBSLAM3R_VOC=/workspace/Vocabulary/ORBvoc.txt
ORBSLAM3R_EX=/workspace/Examples
ORBSLAM3R_KITTI=/datasets/kitti_dataset/data_odometry_gray/dataset/sequences
ORBSLAM3R_TUM=/datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk

# KITTI ships 00-21 but only 00-10 carry ground-truth poses; 11-21 are the
# held-out evaluation split and cannot be scored.
dataset_sequences() {
  case "$1" in
    euroc) echo "MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203" ;;
    kitti) echo "00 01 02 03 04 05 06 07 08 09 10" ;;
    tum)   echo "fr1desk" ;;
    *) return 1 ;;
  esac
}

dataset_configs() {
  case "$1" in
    euroc) echo "mono stereo mono_inertial stereo_inertial" ;;
    kitti) echo "mono stereo" ;;
    tum)   echo "mono rgbd" ;;
    *) return 1 ;;
  esac
}

_kitti_yaml() {
  case "$1" in
    00|01|02) echo KITTI00-02.yaml ;;
    03)       echo KITTI03.yaml ;;
    *)        echo KITTI04-12.yaml ;;
  esac
}

# One line for one (dataset, config, sequence). The trailing argument on the
# EuRoC binaries is the trajectory basename: they write f_<name>.txt and
# kf_<name>.txt into the working directory. The KITTI and TUM binaries have no
# such argument and always write CameraTrajectory.txt / KeyFrameTrajectory.txt,
# which is why every caller must give each run its own working directory.
_plan_line() {
  local ds="$1" cfg="$2" seq="$3"
  local V="$ORBSLAM3R_VOC" E="$ORBSLAM3R_EX"
  case "${ds}_${cfg}" in
    euroc_mono)
      echo "euroc_${seq}_mono|Monocular/mono_euroc|$V $E/Monocular/EuRoC.yaml /datasets/EuRoC/$seq $E/Monocular/EuRoC_TimeStamps/$seq.txt t" ;;
    euroc_stereo)
      echo "euroc_${seq}_stereo|Stereo/stereo_euroc|$V $E/Stereo/EuRoC.yaml /datasets/EuRoC/$seq $E/Stereo/EuRoC_TimeStamps/$seq.txt t" ;;
    euroc_mono_inertial)
      echo "euroc_${seq}_mono_inertial|Monocular-Inertial/mono_inertial_euroc|$V $E/Monocular-Inertial/EuRoC.yaml /datasets/EuRoC/$seq $E/Monocular-Inertial/EuRoC_TimeStamps/$seq.txt t" ;;
    euroc_stereo_inertial)
      echo "euroc_${seq}_stereo_inertial|Stereo-Inertial/stereo_inertial_euroc|$V $E/Stereo-Inertial/EuRoC.yaml /datasets/EuRoC/$seq $E/Stereo-Inertial/EuRoC_TimeStamps/$seq.txt t" ;;
    kitti_mono)
      echo "kitti_${seq}_mono|Monocular/mono_kitti|$V $E/Monocular/$(_kitti_yaml "$seq") $ORBSLAM3R_KITTI/$seq" ;;
    kitti_stereo)
      echo "kitti_${seq}_stereo|Stereo/stereo_kitti|$V $E/Stereo/$(_kitti_yaml "$seq") $ORBSLAM3R_KITTI/$seq" ;;
    tum_mono)
      echo "tum_fr1desk_mono|Monocular/mono_tum|$V $E/Monocular/TUM1.yaml $ORBSLAM3R_TUM" ;;
    tum_rgbd)
      echo "tum_fr1desk_rgbd|RGB-D/rgbd_tum|$V $E/RGB-D/TUM1.yaml $ORBSLAM3R_TUM $E/RGB-D/associations/fr1_desk.txt" ;;
    *) return 1 ;;
  esac
}

dataset_plan() {
  local want_ds=all want_cfg=all
  if [ $# -gt 0 ]; then want_ds="$1"; shift; fi
  if [ $# -gt 0 ]; then want_cfg="$1"; shift; fi
  # bash 3.2 (the macOS system shell) treats "${a[@]}" on an empty array as an
  # unbound variable under `set -u`, hence the guard.
  local seqs=("${@+$@}")

  local ds
  for ds in euroc kitti tum; do
    [ "$want_ds" = all ] || [ "$want_ds" = "$ds" ] || continue

    local cfgs; cfgs=$(dataset_configs "$ds")
    local these=("${seqs[@]+"${seqs[@]}"}")
    [ ${#these[@]} -gt 0 ] || read -r -a these <<< "$(dataset_sequences "$ds")"

    local cfg seq
    for seq in "${these[@]}"; do
      for cfg in $cfgs; do
        [ "$want_cfg" = all ] || [ "$want_cfg" = "$cfg" ] || continue
        _plan_line "$ds" "$cfg" "$seq" || true
      done
    done
  done
}
