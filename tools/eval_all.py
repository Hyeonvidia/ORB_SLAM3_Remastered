#!/usr/bin/env python3
"""
Scores every run in results/matrix against the right ground truth, and prints
one table per dataset.

Run inside the dev container (numpy and /datasets live there):

  ./docker/run.sh -- python3 /workspace/tools/eval_all.py

Three choices per run are not interchangeable, and getting any of them wrong
quietly changes the number:

  GROUND-TRUTH FRAME
    ORB-SLAM3 reports pure-visual trajectories in the LEFT CAMERA frame and
    visual-inertial ones in the IMU BODY frame, and ships a separate transformed
    ground truth for the first case. Scoring an inertial run against the
    left-camera reference leaves a camera-to-IMU offset that rotates with the
    trajectory, so no single global alignment can cancel it -- on EuRoC MH01
    stereo-inertial that alone moved ATE from 4.4 cm to 7.5 cm.

  ALIGNMENT GROUP
    Only monocular is scale-free, so only monocular aligns with Sim(3). Stereo,
    RGB-D and inertial configurations observe metric scale; aligning them with
    Sim(3) would absorb a real scale error into the fit. Those are aligned with
    SE(3), and the Sim(3) scale is reported separately as a diagnostic.

  ASSOCIATION
    EuRoC and TUM carry timestamps on both sides. KITTI ground truth has none:
    it is one 3x4 pose per frame, matched to stereo_kitti's identically shaped
    output by index, and to mono_kitti's keyframe timestamps through the
    sequence's times.txt.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace")
MATRIX = ROOT / "results/matrix"
DATASETS = Path("/datasets")
EVAL = ROOT / "tools/evaluate_ate.py"

EUROC_SEQS = ["MH01", "MH02", "MH03", "MH04", "MH05",
              "V101", "V102", "V103", "V201", "V202", "V203"]
KITTI_SEQS = [f"{i:02d}" for i in range(11)]
KITTI_GRAY = DATASETS / "kitti_dataset/data_odometry_gray/dataset/sequences"
KITTI_POSES = DATASETS / "kitti_dataset/data_odometry_poses/dataset/poses"
TUM_SEQ = DATASETS / "TUM_RGBD/rgbd_dataset_freiburg1_desk"


def score(estimate, groundtruth, scale=False, times=None, gt_times=None):
    """-> (rmse, pairs, sim3_scale, length_m) or None if the run produced nothing."""
    if not Path(estimate).is_file() or not Path(groundtruth).is_file():
        return None
    cmd = [sys.executable, str(EVAL), str(estimate), str(groundtruth)]
    if scale:
        cmd.append("--scale")
    if times:
        cmd += ["--times", str(times)]
    if gt_times:
        cmd += ["--gt-times", str(gt_times)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    rmse = pairs = sim3 = length = None
    for line in out.stdout.splitlines():
        # Values carry a unit suffix ("0.0232 m"), so take the first token.
        if line.startswith("ATE RMSE"):
            rmse = float(line.split(":")[1].split()[0])
        elif line.startswith("matched pairs"):
            pairs = int(line.split(":")[1].split()[0])
        elif line.startswith("scale "):
            sim3 = float(line.split(":")[1].split()[0])
        elif line.startswith("trajectory length"):
            length = float(line.split(":")[1].split()[0])
    return (rmse, pairs, sim3, length) if rmse is not None else None


def report(title, rows):
    # Absolute ATE alone does not compare across datasets -- EuRoC paths are
    # tens of metres, KITTI's are kilometres -- so the path length and the
    # error as a fraction of it are shown alongside.
    print(f"\n{title}")
    print(f"{'RUN':<24}{'ATE RMSE':>10}{'PATH':>10}{'ATE/PATH':>10}  {'PAIRS':>6}  NOTE")
    print(f"{'-' * 22:<24}{'-' * 8:>10}{'-' * 8:>10}{'-' * 8:>10}  {'-' * 5:>6}  {'-' * 22}")
    scored, ratios = [], []
    for name, result, note in rows:
        if result is None:
            print(f"{name:<24}{'-':>10}{'-':>10}{'-':>10}  {'-':>6}  not scored")
            continue
        rmse, pairs, _, length = result
        scored.append(rmse)
        ratio = f"{100 * rmse / length:>9.3f}%" if length else f"{'-':>10}"
        if length:
            ratios.append(100 * rmse / length)
        path = f"{length:>9.1f}m" if length else f"{'-':>10}"
        print(f"{name:<24}{rmse:>10.4f}{path}{ratio}  {pairs:>6}  {note}")
    if scored:
        med = sorted(scored)[len(scored) // 2]
        med_r = sorted(ratios)[len(ratios) // 2] if ratios else None
        print(f"{'':<24}{'':>10}{'':>10}{'':>10}")
        tail = f"{med_r:>9.3f}%" if med_r is not None else ""
        print(f"{'median of ' + str(len(scored)):<24}{med:>9.4f}m{'':>10}{tail}")
    return scored


def euroc():
    rows = []
    cam_gt = ROOT / "evaluation/Ground_truth/EuRoC_left_cam"
    for seq in EUROC_SEQS:
        imu_gt = DATASETS / f"EuRoC/{seq}/mav0/state_groundtruth_estimate0/data.csv"
        for cfg, gt, use_scale in [
            ("mono", cam_gt / f"{seq}_GT.txt", True),
            ("stereo", cam_gt / f"{seq}_GT.txt", False),
            ("mono_inertial", imu_gt, False),
            ("stereo_inertial", imu_gt, False),
        ]:
            traj = MATRIX / f"euroc_{seq}_{cfg}" / "f_t.txt"
            result = score(traj, gt, scale=use_scale)
            note = "Sim(3)" if use_scale else "SE(3)"
            if not use_scale and result:
                s = score(traj, gt, scale=True)
                if s and s[2]:
                    note += f", scale err {100 * (s[2] - 1):+.2f}%"
            rows.append((f"{seq} {cfg}", result, note))
    return report("EuRoC  (11 sequences x 4 configurations)", rows)


def kitti():
    rows = []
    for seq in KITTI_SEQS:
        gt = KITTI_POSES / f"{seq}.txt"
        times = KITTI_GRAY / seq / "times.txt"

        # mono_kitti writes keyframes in TUM format, timestamped from times.txt;
        # the ground truth needs the same clock attached.
        rows.append((
            f"{seq} mono",
            score(MATRIX / f"kitti_{seq}_mono" / "KeyFrameTrajectory.txt", gt,
                  scale=True, gt_times=times),
            "Sim(3), keyframes",
        ))
        # stereo_kitti writes one 3x4 pose per frame, same shape as the ground
        # truth, so they line up by index.
        rows.append((
            f"{seq} stereo",
            score(MATRIX / f"kitti_{seq}_stereo" / "CameraTrajectory.txt", gt),
            "SE(3), per frame",
        ))
    return report("KITTI odometry  (sequences 00-10, the ones with ground truth)", rows)


def tum():
    gt = TUM_SEQ / "groundtruth.txt"
    rows = [
        ("fr1_desk mono",
         score(MATRIX / "tum_fr1desk_mono" / "KeyFrameTrajectory.txt", gt, scale=True),
         "Sim(3), keyframes"),
        ("fr1_desk rgbd",
         score(MATRIX / "tum_fr1desk_rgbd" / "CameraTrajectory.txt", gt),
         "SE(3), per frame"),
        ("fr1_desk rgbd (kf)",
         score(MATRIX / "tum_fr1desk_rgbd" / "KeyFrameTrajectory.txt", gt),
         "SE(3), keyframes"),
    ]
    return report("TUM RGB-D  (freiburg1_desk)", rows)


def main():
    if not MATRIX.is_dir():
        sys.exit(f"missing {MATRIX} -- run tools/run_all.sh first")
    euroc()
    kitti()
    tum()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
