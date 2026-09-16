#!/usr/bin/env python3
"""
Builds a TUM RGB-D association file: pairs each RGB frame with the depth frame
closest in time.

ORB-SLAM3's rgbd_tum example needs one, but the TUM archives do not ship it --
upstream points at the benchmark's associate.py, which is Python 2 and will not
run on a current distribution.

  ./tools/associate_tum.py <sequence_dir> [out.txt]

Output lines: "<rgb_ts> <rgb_path> <depth_ts> <depth_path>", which is the order
rgbd_tum's LoadImages expects.
"""
import sys
from pathlib import Path

MAX_DIFFERENCE = 0.02   # seconds; the benchmark's own default


def read_index(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append((float(parts[0]), parts[1]))
    return sorted(rows)


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <sequence_dir> [out.txt]")
    seq = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else seq / "associations.txt"

    rgb = read_index(seq / "rgb.txt")
    depth = read_index(seq / "depth.txt")
    if not rgb or not depth:
        sys.exit(f"missing or empty rgb.txt / depth.txt under {seq}")

    # Greedy nearest match, each depth frame consumed at most once.
    used = set()
    lines = []
    for t_rgb, p_rgb in rgb:
        best_i, best_d = None, MAX_DIFFERENCE
        for i, (t_d, _) in enumerate(depth):
            if i in used:
                continue
            d = abs(t_d - t_rgb)
            if d < best_d:
                best_i, best_d = i, d
            elif t_d > t_rgb + MAX_DIFFERENCE:
                break
        if best_i is not None:
            used.add(best_i)
            t_d, p_d = depth[best_i]
            lines.append(f"{t_rgb:.6f} {p_rgb} {t_d:.6f} {p_d}")

    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} pairs from {len(rgb)} rgb / {len(depth)} depth")


if __name__ == "__main__":
    main()
