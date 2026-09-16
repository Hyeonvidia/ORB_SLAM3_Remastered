#!/usr/bin/env python3
"""
Absolute Trajectory Error against ground truth, with optional scale alignment.

Replaces ORB-SLAM3's evaluation/evaluate_ate_scale.py, which is Python 2 and
will not run on any current distribution.

Method: associate estimate and ground-truth poses by nearest timestamp, align
them with Umeyama's closed-form similarity transform, and report the RMS of the
residual translation error.

  --scale   solve for scale too (7-DoF).  Required for monocular SLAM, whose
            trajectory is only determined up to scale; using 6-DoF there
            reports the scale error rather than the trajectory error.

  ./tools/evaluate_ate.py estimate.txt groundtruth.txt --scale

Both files are whitespace- or comma-separated `timestamp tx ty tz ...`; any
further columns (quaternion, covariance) are ignored, and `#` comments are
skipped.  Timestamps in nanoseconds are detected and converted.
"""
import argparse
import sys

import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            rows.append([float(x) for x in parts[:4]])
    if not rows:
        sys.exit(f"no poses in {path}")
    data = np.asarray(rows)
    t, xyz = data[:, 0], data[:, 1:4]
    # EuRoC writes nanoseconds; anything this large is not seconds.
    if np.median(t) > 1e12:
        t = t * 1e-9
    return t, xyz


def associate(t_est, t_gt, max_difference):
    """Nearest-timestamp matching, each ground-truth pose used at most once."""
    order = np.argsort(t_gt)
    t_gt_sorted = t_gt[order]
    idx = np.searchsorted(t_gt_sorted, t_est)
    idx = np.clip(idx, 1, len(t_gt_sorted) - 1)
    left, right = t_gt_sorted[idx - 1], t_gt_sorted[idx]
    pick = np.where(np.abs(t_est - left) <= np.abs(t_est - right), idx - 1, idx)

    pairs, used = [], set()
    for i, j in enumerate(pick):
        if abs(t_est[i] - t_gt_sorted[j]) > max_difference:
            continue
        gt_index = order[j]
        if gt_index in used:
            continue
        used.add(gt_index)
        pairs.append((i, gt_index))
    return pairs


def umeyama(src, dst, with_scale):
    """Similarity transform taking src onto dst (Umeyama 1991).

    Returns (R, t, s) minimising sum ||s*R*src_i + t - dst_i||^2.
    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    # Guards against the reflection SVD can return when the points are coplanar.
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / len(src)
        s = float(np.trace(np.diag(D) @ S) / var_src)
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate")
    ap.add_argument("groundtruth")
    ap.add_argument("--scale", action="store_true",
                    help="also solve for scale (use for monocular)")
    ap.add_argument("--max-difference", type=float, default=0.02,
                    help="association window in seconds (default 0.02)")
    ap.add_argument("--save-aligned", metavar="PATH",
                    help="write the aligned estimate for plotting")
    args = ap.parse_args()

    t_est, xyz_est = load(args.estimate)
    t_gt, xyz_gt = load(args.groundtruth)
    pairs = associate(t_est, t_gt, args.max_difference)
    if len(pairs) < 3:
        sys.exit(f"only {len(pairs)} pose pairs matched; nothing to evaluate")

    src = xyz_est[[i for i, _ in pairs]]
    dst = xyz_gt[[j for _, j in pairs]]
    R, t, s = umeyama(src, dst, args.scale)

    aligned = (s * (R @ src.T)).T + t
    err = np.linalg.norm(aligned - dst, axis=1)

    print(f"estimate            : {args.estimate}")
    print(f"ground truth        : {args.groundtruth}")
    print(f"estimate poses      : {len(t_est)}")
    print(f"ground-truth poses  : {len(t_gt)}")
    print(f"matched pairs       : {len(pairs)}")
    print(f"alignment           : {'Sim(3), 7-DoF' if args.scale else 'SE(3), 6-DoF'}")
    if args.scale:
        print(f"scale               : {s:.6f}")
    print(f"trajectory length   : {np.linalg.norm(np.diff(dst, axis=0), axis=1).sum():.3f} m")
    print()
    print(f"ATE RMSE            : {np.sqrt((err ** 2).mean()):.6f} m")
    print(f"ATE mean            : {err.mean():.6f} m")
    print(f"ATE median          : {np.median(err):.6f} m")
    print(f"ATE max             : {err.max():.6f} m")

    if args.save_aligned:
        np.savetxt(args.save_aligned,
                   np.column_stack([t_est[[i for i, _ in pairs]], aligned]),
                   fmt="%.9f")
        print(f"\naligned estimate -> {args.save_aligned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
