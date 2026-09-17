#!/usr/bin/env python3
"""
Moves the SLAM sources into folders that match the architecture in the
ORB-SLAM3 paper, and rewrites the includes to match.

The layers come from Figure 1 of reference/Related Publications/ORB-SLAM3.pdf,
which draws the system as an Atlas plus four threads:

  atlas/          the multi-map representation -- active and non-active maps,
                  each with its MapPoints, KeyFrames, covisibility graph and
                  spanning tree -- and the DBoW2 keyframe database that holds
                  the visual vocabulary and the recognition database
  tracking/       per-frame pose estimation, relocalisation, map initialisation
  local_mapping/  keyframe insertion, point culling and creation, local BA,
                  IMU initialisation and scale refinement
  loop_closing/   place recognition, loop correction and map merging
  optimization/   the g2o problems every thread builds, including Full BA

and three supporting layers the paper treats separately or not at all:

  camera/         Section IV: the camera model is deliberately abstracted out
                  of the pipeline, with pin-hole and Kannala-Brandt behind it
  features/       ORB extraction and matching
  common/         conversions, geometry, settings, IMU types, logging
  viewer/         not part of the algorithm

After this, an include line says which layer it reaches into, so a dependency
that crosses the architecture is visible at the point of use.

  ./tools/layer_layout.py --list
  ./tools/layer_layout.py
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

LAYERS = {
    "atlas": ["Atlas", "Map", "MapPoint", "KeyFrame", "KeyFrameDatabase", "ORBVocabulary"],
    "tracking": ["Tracking", "Frame", "MLPnPsolver"],
    "local_mapping": ["LocalMapping"],
    "loop_closing": ["LoopClosing", "Sim3Solver"],
    "optimization": ["Optimizer", "G2oTypes", "OptimizableTypes"],
    "camera": ["GeometricCamera", "Pinhole", "KannalaBrandt8"],
    "features": ["ORBextractor", "ORBmatcher"],
    "common": ["Converter", "GeometricTools", "Settings", "Verbose",
               "SerializationUtils", "ImuTypes", "TwoViewReconstruction"],
    "viewer": ["Viewer", "FrameDrawer", "MapDrawer"],
}
# System sits above the layers and stays at the root of each tree.
UNMOVED = {"System"}


def unit_layer():
    out = {}
    for layer, units in LAYERS.items():
        for u in units:
            out[u] = layer
    return out


def find(stem, ext, base):
    hits = [p for p in (ROOT / base).rglob(f"{stem}.{ext}")]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    mapping = unit_layer()
    moves = []
    for stem, layer in sorted(mapping.items()):
        for base, ext in (("include", "hpp"), ("src", "cpp")):
            src = find(stem, ext, base)
            if not src:
                continue
            dst = ROOT / base / layer / f"{stem}.{ext}"
            if src != dst:
                moves.append((src, dst))

    for src, dst in moves:
        print(f"{src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")
    print(f"\n{len(moves)} files")
    if args.list:
        return 0

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=False)
        if src.exists():                      # not tracked by git
            shutil.move(str(src), str(dst))

    # Rewrite every include of a moved header to carry its layer.
    edited = 0
    targets = [p for base in ("include", "src", "Examples")
               for p in (ROOT / base).rglob("*")
               if p.is_file() and p.suffix in {".hpp", ".cpp"}]
    for path in targets:
        text = original = path.read_text(errors="replace")
        for stem, layer in mapping.items():
            text = re.sub(r'#include\s*"(?:[A-Za-z0-9_]+/)?' + stem + r'\.hpp"',
                          f'#include "{layer}/{stem}.hpp"', text)
        if text != original:
            path.write_text(text)
            edited += 1
    print(f"rewrote includes in {edited} files")

    # Drop the directories the old layout left behind.
    old = ROOT / "include" / "CameraModels"
    if old.is_dir() and not any(old.iterdir()):
        old.rmdir()
    old = ROOT / "src" / "CameraModels"
    if old.is_dir() and not any(old.iterdir()):
        old.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
