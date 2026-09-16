# Dependencies

This replaces upstream ORB-SLAM3's `Dependencies.md`. That file was written by
hand and has drifted from the code; every claim below was instead **measured**
against upstream git history by `tools/classify_vendored.sh` and
`tools/upstream_delta.py`.

## How dependencies are managed here

Each borrowed library is a **git submodule pinned to an upstream release tag**,
checked out byte-for-byte as upstream published it. ORB-SLAM3's changes to those
libraries are not applied to the checkouts — they live beside them in
`vendor_ext/`. See [WRAPPERS.md](WRAPPERS.md).

| Submodule | Upstream | Pinned tag | Role |
|---|---|---|---|
| `thirdparty/g2o` | RainerKuemmerle/g2o | `20241228_git` | graph optimization |
| `thirdparty/Sophus` | strasdat/Sophus | `1.24.6` | Lie groups (header-only) |
| `thirdparty/DBoW2` | dorian3d/DBoW2 | `v1.1-free` | place recognition |
| `thirdparty/DLib` | dorian3d/DLib | `v1.1-free` | `DUtils::Random`, required by DBoW2 |
| `thirdparty/Pangolin` | stevenlovegrove/Pangolin | `v0.9.6` | viewer / GL |
| `reference/ORB_SLAM3` | UZ-SLAMLab/ORB_SLAM3 | `v1.0-release` | diff reference, never built |

Re-pin with `tools/init_submodules.sh`, then `tools/gen_manifest.sh` and rebuild
the `thirdparty` image. The image carries the manifest at
`/opt/orbslam3r/THIRDPARTY_MANIFEST.txt`, so a running container can always say
which revisions it was built from.

System packages (from Ubuntu 24.04, not submodules): OpenCV 4.6.0,
Eigen 3.4.0, Boost.Serialization 1.83, SuiteSparse, spdlog/fmt, Mesa.

## What ORB-SLAM3 actually changed in each borrowed library

Both tools compare against **upstream history**, not against the current tip.
A plain diff of an eight-year-old fork against a 2024 release attributes every
upstream commit since to ORB-SLAM3 — for g2o that is ~15,000 lines of noise.
`classify_vendored.sh` tests blob identity (did this exact file ever exist
upstream?); `upstream_delta.py` finds each file's closest upstream ancestor and
reports only the residual, with CRLF folded so line-ending churn is not counted.

| Library | Files changed | Lines | What the change is |
|---|---:|---:|---|
| **Sophus** | **0 / 21** | **0** | Nothing. Purely vendored. |
| **DBoW2** | 7 / 14 | 237 | Text vocabulary format (139), Boost serialization (20), `FORB` signature tweaks (75), 3 lines upstream has since fixed |
| **g2o** | 63 / 95 | 1105 | Almost entirely include-path rewrites for the stripped tree, plus fork drift. One file with no upstream counterpart (`types/se3mat.*`) is dead code. |

Per-file diffs: `docs/modifications/<project>/`. Regenerate any time with
`./tools/upstream_delta.py --project <g2o|Sophus|DBoW2> --out docs/modifications`.

### Sophus — the upstream doc is wrong

`Dependencies.md` upstream states Sophus is *"a modified version of Sophus"*.
It is not. All 21 headers in `Thirdparty/Sophus/sophus` match blobs in upstream
history exactly; ORB-SLAM3 froze an untagged 1.1.0-era master snapshot and
changed nothing. This project therefore uses upstream `1.24.6` directly, with no
wrapper of any kind.

### g2o — the unique addition is dead code

`types/se3mat.h` / `.cpp` is the only vendored g2o file with no upstream
counterpart anywhere. Nothing in ORB-SLAM3's `src/` or `include/` references
`SE3mat`, and its `ExpSO3`/`LogSO3` duplicate the ones ORB-SLAM3 defines in
[`include/G2oTypes.h`](../reference/ORB_SLAM3/include/G2oTypes.h). It is not
carried forward.

Of the 26 distinct `g2o::` symbols ORB-SLAM3 names, upstream `20241228_git`
provides 24 under the same name. The two exceptions are renames, handled by two
`using` declarations in `vendor_ext/g2o_ext/compat.hpp`:

| ORB-SLAM3 | upstream 20241228 | uses in ORB-SLAM3 |
|---|---|---:|
| `g2o::VertexSBAPointXYZ` | `g2o::VertexPointXYZ` | 66 |
| `g2o::Vector7d` | `g2o::Vector7` | 3 |

## Single-file borrowings

These are forks of one source file rather than of a library, so there is no
submodule to pin. Upstream ORB-SLAM3 lists them in its `Dependencies.md`.

| ORB-SLAM3 file | Origin | License | Status |
|---|---|---|---|
| `src/features/ORBextractor.cpp` | OpenCV **2.4.x** `modules/features2d/src/orb.cpp` | BSD | Fork. See below. |
| `src/tracking/MLPnPsolver.cpp` | Steffen Urban's MLPnP, via OpenGV | BSD | Fork. |
| `ORBmatcher::DescriptorDistance` | Stanford "Bit Twiddling Hacks" parallel popcount | public domain | Snippet. `__builtin_popcount` is the modern equivalent. |
| ~~`PnPsolver.h/.cc`~~ | Lepetit's EPnP | FreeBSD | **Does not exist in v1.0.** Removed when MLPnP replaced it; upstream's `Dependencies.md` still lists it. |

### ORBextractor: which OpenCV it came from, and why it is not split

The ancestor is **OpenCV 2.4.x**, not the 4.6.0 this project builds against.
That distinction is the whole reason the file looks unrecognisable next to
today's OpenCV: 3.0 rewrote this code, folding the angle and descriptor loops
into `ICAngles` and `computeOrbDescriptors`. Diffing against 4.6.0 therefore
reports *every* symbol as rewritten, which is an artefact of comparing against
the wrong ancestor, not a measure of what ORB-SLAM3 changed.

Measured against 2.4.13.7, roughly **442 of 1022 lines** are OpenCV-derived --
including `ComputePyramid` and the descriptor scaffolding inside `operator()`,
which are easy to mistake for ORB-SLAM3 originals. What ORB-SLAM3 genuinely
added is the octree keypoint distribution (`DistributeOctTree`,
`ComputeKeyPointsOctTree`), which spreads features evenly over the image
instead of taking the strongest N.

**It is deliberately not split into an "OpenCV part" and an "ORB-SLAM3 part".**
The two are interleaved at the level of individual loops, so any cut leaves
OpenCV-derived code sitting in a file labelled ORB-SLAM3 original -- a *worse*
licence statement than the honest dual notice the file carries today. And there
is nothing to fix on the compliance side: BSD-3 inside GPLv3 only obliges us to
retain the notice, which `src/features/ORBextractor.cpp` already does at the
top of the file.

## Licenses

ORB-SLAM3 itself is GPLv3. Everything borrowed is permissive:

- OpenCV, DBoW2, DLib, g2o, EPnP, MLPnP — BSD family
- Sophus, Pangolin — MIT
- Eigen ≥ 3.1.1 — MPL2
- `DescriptorDistance` bit-twiddling — public domain

Commercial licensing of ORB-SLAM3 goes through the authors: orbslam (at) unizar (dot) es.
