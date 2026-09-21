# Porting the SLAM body

`src/`, `include/` and `Examples/` are **generated** from the pristine
ORB-SLAM3 v1.0 checkout in `reference/`. Nothing there is hand-edited.

```bash
./tools/port_all.sh --force
```

Three stages, each of which explains itself in its own header comment:

| Stage | Script | What it does |
|---|---|---|
| 1 | `port_body.py` | Rewrites the include graph from the bundled `Thirdparty/` tree onto the pinned upstream packages and the `vendor_ext/` wrappers. Normalises extensions: `.cc` → `.cpp`, `.h` → `.hpp`. |
| 2 | `qualify_std.py` | Adds `std::` qualification and the standard headers the code had been getting through a leaked `using namespace std`. |
| 3 | `port_fixes.py` | Targeted source fixes the first two cannot express. |

Keeping the port scripted rather than hand-applied means it replays against a
different upstream pin, and that "what does the remaster change?" is answered by
reading three files instead of diffing 36,000 lines.

**Put new changes in `port_fixes.py`, not in `src/`** — `port_all.sh` overwrites
the tree.

## Stage 1 — the include graph and file extensions

### Extensions

ORB-SLAM3 ships 24 sources as `.cc` and 2 as `.cpp`, and all 31 headers as `.h`.
The tree now uses `.cpp` and `.hpp` throughout, which also matches `vendor_ext/`
— otherwise one project would carry two conventions side by side.

Only headers under `include/` are renamed, and only includes that name one of
them are rewritten. External headers keep their own spelling: `<DBoW2/FORB.h>`,
`<g2o/core/block_solver.h>` and `<DUtils/Random.h>` are untouched, because they
belong to upstream packages this project does not rename.

Includes are matched under all three spellings ORB-SLAM3 uses for its own
headers — `"Frame.h"`, `"CameraModels/Pinhole.h"` and `"include/CameraModels/Pinhole.h"` —
since the build puts both `include/` and `include/CameraModels` on the search
path.

### Includes

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

### `mnFullBAIdx` is incremented and compared, but declared `bool`

Three sites in `LoopClosing` abort a running global bundle adjustment:

```cpp
mbStopGBA = true;
mnFullBAIdx++;
```

and `RunGlobalBundleAdjustment` guards its map update with:

```cpp
int idx = mnFullBAIdx;
{
    std::unique_lock<std::mutex> lock(mMutexGBA);
    if (idx != mnFullBAIdx) return;   // superseded — discard this result
    ...
}
```

The member is `bool mnFullBAIdx;`. The first `++` sets it to `true`; every later
one is a no-op. So after the first abort in a session, `idx != mnFullBAIdx` can
never be true again and that guard is permanently dead. ORB-SLAM2 declared the
same member `int`.

C++17 removing `operator++` on `bool` is the only reason this surfaced at all —
before that it compiled with a deprecation warning nobody reads.

**The fix is one word**, in `include/LoopClosing.h`:

```diff
-    bool mnFullBAIdx;
+    // Counter, not a flag: incremented on every GBA abort and compared
+    // with != in RunGlobalBundleAdjustment. As a bool it saturates at
+    // true and that comparison stops working.
+    int mnFullBAIdx;
```

**Scope.** Only the type changed. ORB-SLAM3 also moved the snapshot to *after*
the optimisation returns — ORB-SLAM2 read it before — so even as an `int` the
guard now only covers the gap between that read and acquiring `mMutexGBA`, which
is microseconds. Restoring the earlier snapshot would change runtime behaviour
rather than fix a compile error, and the primary abort handling is `mbStopGBA`,
checked separately a few lines below. So this is a correctness fix to a
secondary race guard, not a fix to how aborts are handled.

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

### One behaviour that was misread

The inertial runs used to log `Cholesky failure, writing debug.txt` from g2o's
Eigen linear solver during visual-inertial BA. This section said the port had
not introduced that -- that the same failure path exists in ORB-SLAM3's fork
and only upstream's logging made it visible. The path exists, and the fork
would have printed the message too; it never did, because it never failed. The
fork factorises with LDLT and upstream, since 2020, with LLT, which refuses the
numerically indefinite Hessians an inertial BA produces. The port did introduce
it, by taking upstream's solver as it came. `vendor_ext`'s
`LinearSolverEigenLDLT` restores the fork's factorisation; see
[WRAPPERS.md](WRAPPERS.md).

## Not ported

- **RealSense examples** — need librealsense2, which has no linux/arm64 package
  in this image and no camera reachable from the container.
- **ROS examples** — the ROS node wrappers under `Examples/ROS` target ROS
  Melodic on Ubuntu 18.04.
