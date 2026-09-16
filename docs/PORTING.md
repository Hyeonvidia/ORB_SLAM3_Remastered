# Porting the SLAM body

`src/`, `include/` and `Examples/` are **generated** from the pristine
ORB-SLAM3 v1.0 checkout in `reference/`. Nothing there is hand-edited.

```bash
./tools/port_all.sh --force
```

Three stages, each of which explains itself in its own header comment:

| Stage | Script | What it does |
|---|---|---|
| 1 | `port_body.py` | Rewrites the include graph from the bundled `Thirdparty/` tree onto the pinned upstream packages and the `vendor_ext/` wrappers. Renames `.cc` → `.cpp`. |
| 2 | `qualify_std.py` | Adds `std::` qualification and the standard headers the code had been getting through a leaked `using namespace std`. |
| 3 | `port_fixes.py` | Targeted source fixes the first two cannot express. |

Keeping the port scripted rather than hand-applied means it replays against a
different upstream pin, and that "what does the remaster change?" is answered by
reading three files instead of diffing 36,000 lines.

**Put new changes in `port_fixes.py`, not in `src/`** — `port_all.sh` overwrites
the tree.

## Stage 1 — the include graph

42 rewrites across 13 files. Most are mechanical (`Thirdparty/g2o/g2o/core/x.h`
→ `g2o/core/x.h`), with three that are not:

- `types/types_six_dof_expmap.h`, `types_seven_dof_expmap.h`, `types_sba.h` and
  `sim3.h` all route through `orbslam3r/g2o_ext/compat.hpp`. Upstream
  reorganised `types/` into `types/sba/`, `types/sim3/` and `types/slam3d/`, and
  the wrapper supplies the two renamed symbols so call sites keep their spelling.
- `solvers/linear_solver_eigen.h` → `g2o/solvers/eigen/linear_solver_eigen.h`.
- `ORBVocabulary.h`'s typedef now names `orbslam3r::ORBVocabulary`, the
  `vendor_ext` subclass that can read `ORBvoc.txt`.

## Stage 2 — `std::` qualification

**2,521 names qualified, 211 standard headers added, 6 `using namespace std;`
removed, 65 files touched.**

ORB-SLAM3's own headers never say `using namespace std;`. They inherited one:
its forked copy of DBoW2 puts a global `using namespace std;` at
`Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h:36`, and `ORBVocabulary.h` drags
that header into most of the system. Upstream DBoW2 has no such line, so
switching to the pristine package left ~2,500 unqualified names — and several
headers that had been getting `<string>` and `<iostream>` transitively.

Re-adding the `using` would have been one line. Qualifying keeps namespace
pollution from leaking through headers into everything downstream.

Names are matched by the shape of their use, because a blanket word list would
turn `int count = 0;` into `int std::count = 0;`:

| Class | Matched | Examples |
|---|---|---|
| templates | only before `<` | `vector<` `unique_lock<` `numeric_limits<` |
| functions | only before `(` | `sort(` `to_string(` `make_pair(` |
| namespaces | only before `::` | `chrono::` `ios_base::` |
| types | anywhere | `string` `mutex` `ofstream` |
| objects | anywhere | `cout` `cerr` `endl` |

Comments, string literals and preprocessor directives are masked out first —
otherwise `#include <vector>` becomes `#include <std::vector>` and prose like
"list of keyframes" gets mangled.

`find` and `count` are deliberately excluded: ORB-SLAM3 declares
`Sim3Solver::find(...)`, and matching before `(` would rewrite the declaration.

## Stage 3 — targeted fixes

### g2o solver ownership — 33 sites

Every optimizer in `Optimizer.cpp` is built the way g2o worked in 2016:

```cpp
g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
```

Current g2o takes `std::unique_ptr` at both levels, so all 33 are compile
errors. They now read:

```cpp
auto* solver = orbslam3r::g2o_ext::MakeLevenberg<
    g2o::BlockSolver_6_3, orbslam3r::g2o_ext::LinearSolver::kEigen>();
```

A side effect worth noting: where the original builds one `solver_ptr` and then
branches to make two algorithms from it (the `bLarge` case in local inertial
BA), only one algorithm is ever used and the other silently leaks its block
solver. Constructing inside each branch removes that.

### `mnFullBAIdx` is a generation counter declared `bool`

`LoopClosing` uses it to notice that a newer global bundle adjustment request
superseded the one in flight:

```cpp
int idx = mnFullBAIdx;          // before launching GBA
...
if (idx != mnFullBAIdx) return; // superseded — discard this result
```

but the member is `bool mnFullBAIdx;`. `mnFullBAIdx++` saturates at `true` after
the first request, so a second interruption during an already-interrupted GBA
goes unnoticed and a stale optimisation result gets merged into the map.
ORB-SLAM2 declared the same member `int`.

C++17 removing `operator++` on `bool` is the only reason this surfaced.

### `COMPILEDWITHC11` guards — 42 blocks

Every timing call in the examples is wrapped in

```cpp
#ifdef COMPILEDWITHC11
    auto t1 = std::chrono::steady_clock::now();
#else
    auto t1 = std::chrono::monotonic_clock::now();
#endif
```

`std::chrono::monotonic_clock` is a pre-standard name that never existed in
C++11 or later, so the `#else` branch cannot compile on any current toolchain —
the macro only ever hid it. The guards are deleted and the live branch kept.

### The viewer is switchable

Upstream's examples disagree about whether to open the Pangolin viewer:
`mono_euroc` passes `false`, `stereo_euroc` passes `true`,
`stereo_inertial_euroc` passes `false`. Inside the container the viewer runs on
llvmpipe over Xvfb, where it drags a batch run to a crawl — `stereo_euroc` on
MH01 made no measurable progress in twelve minutes with the viewer on, against
3m32s for the whole sequence with it off. (Upstream hit this too: the line under
the check is their own commented-out `if(false) // TODO`.)

`System` now honours `ORBSLAM3R_VIEWER=0`, and the dev container's entrypoint
sets that by default in headless mode. Unset, the caller's flag wins and
behaviour is unchanged; `ORBSLAM3R_VIEWER=1` turns it back on.

### DBoW2 serialization

`KeyFrame.h` archives `DBoW2::BowVector` and `FeatureVector`. ORB-SLAM3 made
that work by adding intrusive `serialize()` members to its forked copies;
`vendor_ext/dbow2_ext/serialization.hpp` supplies the same thing
non-intrusively, and the include is added where the archive is instantiated.

## Result

Builds clean on GCC 13.3 / C++17 — `libORB_SLAM3.so` plus 12 dataset example
binaries, zero errors and zero undefined symbols.

All four sensor configurations run EuRoC MH01 end to end in the container:

| Configuration | ATE RMSE | Note | ORB-SLAM3 paper, MH01 |
|---|---:|---|---:|
| monocular | **0.0170 m** | Sim(3) aligned; scale is a free parameter | ~0.016 m |
| stereo | **0.0351 m** | SE(3) | ~0.035 m |
| monocular-inertial | **0.0841 m** | SE(3), residual scale error 2.18% | ~0.062 m |
| stereo-inertial | **0.0436 m** | SE(3), residual scale error 0.84% | ~0.037 m |

Single runs. ORB-SLAM3's published figures are medians over several executions,
because IMU initialisation and RANSAC make each run different — so treat these
as "the right order of magnitude", not as a reproduction of the table.

```bash
./tools/run_euroc.sh stereo-inertial MH01   # one sequence, one configuration
./tools/eval_euroc.sh MH01                  # score everything in results/
```

### Two evaluation traps

**Ground truth frame.** ORB-SLAM3 reports pure-visual trajectories in the *left
camera* frame and visual-inertial ones in the *IMU body* frame, and ships a
separate transformed ground truth for the first case. Scoring an inertial run
against the left-camera ground truth leaves a camera-to-IMU offset that rotates
with the trajectory, so a single global alignment cannot cancel it — on MH01
stereo-inertial that alone moved ATE from 4.4 cm to 7.5 cm.

**Alignment group.** Only monocular is scale-free. Stereo and inertial
configurations observe metric scale, so aligning them with Sim(3) absorbs a real
scale error into the fit: mono-inertial on MH01 reports 3.0 cm under Sim(3)
against 8.4 cm under SE(3), the difference being a 2.18% scale error that the
first number hides. `tools/eval_euroc.sh` picks the right group per
configuration and reports the scale error separately.

`tools/evaluate_ate.py` replaces `evaluation/evaluate_ate_scale.py`, which is
Python 2 and will not run on any current distribution.

### One behaviour worth not misreading

The inertial runs log `Cholesky failure, writing debug.txt` from g2o's Eigen
linear solver during visual-inertial BA. That is not something the port
introduced: the identical failure path exists in ORB-SLAM3's own g2o fork at
`Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h:107`. Current g2o routes it
through spdlog, which is the only reason it is now visible in the log.

## Not ported

- **RealSense examples** — need librealsense2, which has no linux/arm64 package
  in this image and no camera reachable from the container.
- **ROS examples** — the ROS node wrappers under `Examples/ROS` target ROS
  Melodic on Ubuntu 18.04.
