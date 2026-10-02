# Development environment

The Docker images, how to build and run, formatting, and how the tree is laid out.
Moved here from the README, which keeps the short version.

## Quick start

```bash
git clone --recurse-submodules <this repo> ORB_SLAM3_Remastered
cd ORB_SLAM3_Remastered
./tools/init_submodules.sh     # if you cloned without --recurse-submodules
./docker/build_images.sh       # base -> thirdparty -> dev  (~10 min cold)
./docker/run.sh                # headless shell in the dev container
```

With the viewer, on macOS:

```bash
open -a XQuartz                # Settings > Security > allow network clients
xhost +localhost
./docker/run.sh --gui
```

## The images

| Image | Contents | Rebuild when |
|---|---|---|
| `orbslam3r/base:24.04` | Ubuntu 24.04, GCC 13.3, CMake 3.28, OpenCV 4.6, Eigen 3.4, Boost, Mesa/X11 | the OS baseline moves |
| `orbslam3r/thirdparty:24.04` | g2o, Sophus, DBoW2, DLib, Pangolin built into `/opt/orbslam3r` | a submodule is re-pinned |
| `orbslam3r/dev:24.04` | debugging tools, entrypoint, ccache | rarely |

The project source is **not** baked in — it is bind-mounted at `/workspace`, so
an edit on the host is live in the container. `../Datasets` mounts read-only at
`/datasets`.

`orbslam3r/thirdparty` carries `/opt/orbslam3r/THIRDPARTY_MANIFEST.txt`, so a
running container can always report which upstream revisions it was built from.

### Display modes

The entrypoint takes `ORBSLAM3R_DISPLAY_MODE`:

- `headless` (default) — starts Xvfb on `:99` and forces software GL, so
  Pangolin-linked binaries run with no display attached.
- `x11` — expects `DISPLAY` from the caller (XQuartz via
  `host.docker.internal:0`).

The Pangolin viewer is off by default in headless mode: software rendering makes
a dataset run unusably slow. Set `ORBSLAM3R_VIEWER=1` to force it on, or use
`./docker/run.sh --gui`.

## Verifying the environment

```bash
./docker/run.sh -- bash -c '
  cmake -S /workspace/tools/smoke_test -B /workspace/build/smoke -G Ninja &&
  cmake --build /workspace/build/smoke -j 8 &&
  /workspace/build/smoke/smoke_test --gl'
```

Compiles and links against every dependency, runs a pose-only bundle adjustment
on modern g2o and checks it recovers the true pose, and opens a GL context.

## Building and running the SLAM system

```bash
./tools/build.sh            # incremental
./tools/build.sh --clean    # from scratch (~1 min on 12 cores)
```

Produces `build/lib/libORB_SLAM3.so` and twelve dataset example binaries under
`build/bin/`. On EuRoC MH01:

```bash
./docker/run.sh -- bash -c '
  cd /workspace/results &&
  /workspace/build/bin/Monocular/mono_euroc \
    /workspace/Vocabulary/ORBvoc.txt \
    /workspace/Examples/Monocular/EuRoC.yaml \
    /datasets/EuRoC/MH01 \
    /workspace/Examples/Monocular/EuRoC_TimeStamps/MH01.txt MH01_mono'

./docker/run.sh -- python3 /workspace/tools/evaluate_ate.py \
  /workspace/results/f_MH01_mono.txt \
  /workspace/evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt --scale
```

Extract the vocabulary once first:
`tar -xzf reference/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary`.

### What the library is made of

`src/CMakeLists.txt` builds the one shared library from parts, each an object
library that names the parts and the packages it uses:

| Part | Sources | Uses |
|---|---|---|
| `common` | `common/` except `Settings.cpp` | — |
| `camera` | `camera/` | `common` |
| `features` | `features/` | `common` |
| `optim` | `optim/` (headers so far) | — |
| `optim_g2o` | `optim_g2o/` | `optim camera common` |
| `slam` | `atlas/ tracking/ optimization/ local_mapping/ loop_closing/`, `common/Settings.cpp` | `common camera features optim` |
| `system` | `System.cpp` | `slam common camera` |
| `viewer` | `viewer/` | `system slam common` |
| `noviewer` | `NoViewer.cpp`, in place of `viewer` | `system slam` |

The five layers in `slam` include each other in both directions and stay one
part until that is undone. CMake cannot hold a part to its line -- every header
is under one `include/` -- so `tools/deps_check.py` does, from the table CMake
writes to `build/part_graph.txt`; `ctest` runs it as `part_graph`.

```bash
./docker/run.sh -- python3 /workspace/tools/deps_check.py          # the report
./docker/run.sh -- cmake -S /workspace -B /workspace/build_noviewer -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DORBSLAM3R_BUILD_VIEWER=OFF         # no Pangolin, no OpenGL
```

Without the viewer, `System` gets drawers that do nothing and no viewer thread
(`src/NoViewer.cpp`); the library then depends on 35 shared objects instead of
62.

### How an optimisation is put together

Every optimisation is three steps -- copy what is needed out of the map, solve,
write back -- and the middle one sees neither the map nor the library that
solves it:

- `optim/` says a problem as plain values and what a solver of it must do;
- `optim_g2o/` solves it with g2o, building the graph ORB-SLAM3 v1.0 built, edge
  for edge. It is the only part that names g2o, and `slam` does not name it:
  `optim::Make…()` is declared in `optim/` and defined here, so which library
  solves is decided when the shared library is linked;
- a task in `optimization/` builds the problem from keyframes, points or a
  frame, runs the rounds where there are rounds -- which observations are in,
  which cost they get -- and applies the result. The functions of `Optimizer`
  are a task each, in the source named after it.

| `Optimizer::` | task | problem and solver in `optim/` |
|---|---|---|
| `PoseOptimization` | `PoseTask` | `PoseProblem`, `PoseSolver` |
| `LocalBundleAdjustment` (a keyframe; a welding window), `BundleAdjustment` | `LocalBaTask`, `WeldingBaTask`, `GlobalBaTask` | `BaProblem`, `BundleAdjuster` |
| `OptimizeSim3` | `Sim3Task` | `Sim3Problem`, `Sim3Solver` |
| `OptimizeEssentialGraph` (after a loop; after a merge) | `EssentialGraphTask`, `MergeGraphTask` | `Sim3GraphProblem`, `Sim3GraphSolver` |
| `OptimizeEssentialGraph4DoF` | `EssentialGraph4DofTask` | `Pose4DofGraphProblem`, `Pose4DofGraphSolver` |
| `PoseInertialOptimizationLastKeyFrame`, `…LastFrame` | `InertialPoseTask` | `InertialPoseProblem`, `InertialPoseSolver` |
| `InertialOptimization` (three) | `InertialAlignmentTask` | `InertialAlignmentProblem`, `InertialAlignmentSolver` |
| `LocalInertialBA`, `FullInertialBA`, `MergeInertialBA` | `LocalInertialBaTask`, `FullInertialBaTask`, `MergeInertialBaTask` | `InertialBaProblem`, `InertialBundleAdjuster` |

A problem's unknowns are laid out in the order of its arrays and its terms
summed in the order of theirs; the tasks sort keyframes and points by id and
give the terms in the order v1.0 added its edges. A solver that keeps to both
gives the same bits as v1.0 for the same input, and the g2o one does.

That was shown rather than argued. Until the tag `m1a-shadow` a build with
`-DORBSLAM3R_OPT_SHADOW=ON` ran each task and v1.0's body of the same function
on the same input and counted the calls whose results differed in any bit:
none, over more than a hundred thousand calls, and for each function a change
put in on purpose was counted. Two paths were never reached by
what is on disk: an observation in the second camera of a pair that is not
rectified, and the inertial alignment with gravity and scale held. v1.0's
bodies and that build are gone from the tree; check out the tag to run it.

Two things it found about v1.0 that are still so. Gauss-Newton leaves an edge
the error its last iteration began with and Levenberg-Marquardt that of its
last trial, and the inliers are judged by those; and Local Mapping judges a
monocular observation of an inertial adjustment by whether its point is within
ten metres, a number Tracking writes while it reads.

## Formatting

```bash
./tools/format.sh            # rewrite in place
./tools/format.sh --check    # CI mode: exit 1 if anything differs
```

`.clang-format` at the root is the whole policy. It runs in the dev container,
which pins clang-format 18.1.3, because clang-format's output changes between
major versions and the layout a file gets must not depend on which machine
touched it. VS Code's C/C++ extension bundles a much newer clang-format — the
config was checked against both and they produce byte-identical output, so
format-on-save and `tools/format.sh` agree.

Two things about it are deliberate and easy to undo by accident:

- **`SortIncludes: Never`.** Not because sorting breaks the build — it does
  not — but because it silently deletes three duplicated `#include` lines, so
  "apply the formatter" would quietly stop being a formatting-only commit. It
  also undoes the include grouping that came out of breaking 37 include cycles.
- **`ReflowComments: false`.** At `true`, clang-format welds the two separate
  GPLv3 copyright notices at the top of every file into one paragraph, which
  changes what a legal notice says.

`reference/` and `thirdparty/` are **not** formatted: one is the pristine
baseline every delta is measured against, the other is pinned submodules. Each
carries a `.clang-format` with `DisableFormat: true`, so neither the script nor
an editor that formats on save can reach them.

Three regions carry `// clang-format off` because no setting expresses them: the
Boost `serialize()` bodies (clang-format reads `ar & x` as a reference
declaration), `MLPnPsolver::mlpnpJacs` (machine-generated symbolic algebra), and
the unrolled ORB descriptor loop.

Blame is noisy across the one commit that applied all this. To skip it:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Layout

```
docker/           three Dockerfiles, compose, build_images.sh, run.sh
thirdparty/       pinned upstream submodules — never edited
vendor_ext/       ORB-SLAM3's changes to them, as wrappers
src/ include/     the SLAM library, in layers that follow the paper's Figure 1:
                  atlas/ tracking/ local_mapping/ loop_closing/ optimization/
                  camera/ features/ common/ viewer/  — plus System at the top
                  (.cpp / .hpp throughout; upstream ships .cc / .h)
Examples/         dataset example binaries — also generated
reference/        pristine ORB-SLAM3 v1.0 — diff reference, never built
tools/            submodule pinning, the port pipeline, ATE evaluation, format.sh,
                  the v1.0 baseline build and the A/B against it
docs/             DEPENDENCIES.md, WRAPPERS.md, PORTING.md, FRAME_KEYFRAME.md,
                  modifications/
```

