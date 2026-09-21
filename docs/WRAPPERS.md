# The wrapper architecture

**Goal:** at any moment, be able to answer *"what did ORB-SLAM3 change in this
borrowed library, and why?"* — without reading an eight-year-old fork diff.

**Method:** upstream stays pristine; ORB-SLAM3's changes live beside it as code
you can read on its own.

```
thirdparty/          pinned upstream submodules — never edited
vendor_ext/          ORB-SLAM3's changes to them, as separate code
reference/ORB_SLAM3/ the original fork, for diffing only — never built
docs/modifications/  generated per-file diffs vs. closest upstream ancestor
```

`tools/check_pristine.sh` verifies at any time that `thirdparty/` is still a
clean upstream checkout sitting on its pinned tag. `tools/classify_vendored.sh`
answers the other direction: which files in ORB-SLAM3's own vendored copies were
ever edited at all.

## Why not just patch files?

A patch file records *what* changed but not *why*, and it rots the moment
upstream moves. A wrapper is ordinary code: it compiles, it is tested, it
carries its rationale in comments, and when upstream changes underneath it, the
compiler says so.

## Two tiers

**Tier A — true wrapper.** Upstream exposes an API good enough to extend from
outside. Everything in `vendor_ext/` is Tier A.

**Tier B — pristine reference plus generated delta.** Upstream is a single
source file that must be forked, because there is no seam to extend through.
`ORBextractor` is the case in point: it derives from OpenCV's `orb.cpp`, but
OpenCV does not expose `ORB_Impl`, so no amount of wrapping avoids the fork.
For these, the pristine original is kept and `tools/upstream_delta.py` produces
the diff — the delta stays inspectable even though the code is a fork.

## What is in `vendor_ext/`

### `dbow2_ext/` — 237 lines of DBoW2 change, re-expressed

| ORB-SLAM3 did | Wrapper does | Why it works |
|---|---|---|
| Added `loadFromTextFile`/`saveToTextFile` to `TemplatedVocabulary.h` (139 lines) | `TextFileVocabulary<T,F>` **derives** from upstream's class | Every member the loader touches (`m_k`, `m_L`, `m_nodes`, `m_words`, `Node`, `createScoringObject`) is `protected`, and `~TemplatedVocabulary` is `virtual` |
| Added `friend class boost::serialization::access` + private `serialize()` to `BowVector.h` and `FeatureVector.h` (20 lines) | Free `serialize()` overloads in `serialization.hpp` | Both types derive **publicly** from `std::map`, so a non-intrusive overload reaches everything; Boost only needs the intrusive form for private state. Identical archive bytes. |
| Changed `FORB::distance` to return `int`, made `FORB::L` extern (75 lines) | Nothing | Upstream's `double` return converts implicitly at every call site; `L` is `32` either way |
| Fixed `<opencv/cv.h>`, added a virtual destructor (3 lines) | Nothing | Upstream `v1.1-free` already carries both fixes |

The text vocabulary format itself:

```
line 1     : k  L  scoring  weighting
lines 2..n : parent_id  is_leaf  d0 d1 ... d31  weight
```

Node ids are implicit — the n-th body line is node n — so a parent always
precedes its children.

**Two defects in the original are fixed in the wrapper:**

1. *A missing file was reported as success.* The guard is
   `if (f.eof()) return false;`, but `eof()` is false on a stream that never
   opened. A wrong path produced an empty vocabulary and a silent, total loss of
   place recognition. Now checked with `is_open()`.
2. *A trailing newline appended a garbage node.* `while (!f.eof())` runs one
   extra iteration after the last line; the original body then grew `m_nodes`
   and read an uninitialised parent id from the empty stream. Blank lines are
   now skipped, and a parent id that does not precede its child is rejected.

Both are covered by `vendor_ext/tests/test_vendor_ext.cpp`.

### `g2o_ext/` — two aliases, one factory, one stop rule and one solver

`compat.hpp` is two `using` declarations. That is the whole distance between
ORB-SLAM3's g2o and upstream `20241228_git` *in names* -- it was once described
here as the whole semantic distance, which the two behavioural wrappers further
down show it was not:

```cpp
using VertexSBAPointXYZ = VertexPointXYZ;   // upstream dropped the "SBA" prefix
using Vector7d          = Vector7;          // upstream spells it VectorN<7>
```

`solver_factory.hpp` handles the one genuine API break. ORB-SLAM3 builds
optimizers with raw owning pointers:

```cpp
auto* linear = new g2o::LinearSolverEigen<BlockSolver_6_3::PoseMatrixType>();
auto* block  = new g2o::BlockSolver_6_3(linear);
optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(block));
```

Current g2o takes `std::unique_ptr` at both levels, so every one of those sites
is a compile error — a dozen in `Optimizer.cc` alone, each spelled slightly
differently. Instead of fixing them one by one:

```cpp
optimizer.setAlgorithm(orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3>());
optimizer.setAlgorithm(orbslam3r::g2o_ext::MakeGaussNewton<
    g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kDense>());
```

When g2o changes its ownership convention again, one header moves.

`levenberg_stop_on_stall.hpp` is the one change ORB-SLAM3 made to g2o's
*behaviour*, and the port missed it at first. The fork's Levenberg-Marquardt
counts an iteration that improves the robust chi2 by less than a thousandth as
bad, and returns `Terminate` after three in a row (`//Stop criterium (Raul)` in
`docs/modifications/g2o/core__optimization_algorithm_levenberg.cpp.diff`).
Upstream runs every iteration it is asked for. Nothing failed without the rule
-- the answers agree to nine digits -- but `PoseOptimization` asks for ten
iterations four times over, twice a frame, and converges in five: tracking was
2-10 % slower than v1.0, all of it inside `optimize()`. `LevenbergStopOnStall`
subclasses upstream's algorithm and returns `Terminate` on the same condition;
`MakeLevenberg` hands it out, so every optimisation in the system has it, as in
v1.0. On a synthetic pose problem the iteration counts match the fork's exactly
(63,000 against 63,000, where plain upstream runs 120,000).

g2o's own `SparseOptimizerTerminateAction` was not used: it stops on the first
small gain rather than the third, and it works by taking over the optimizer's
force-stop flag, which Local Mapping already uses to abort a bundle adjustment.

`linear_solver_eigen_ldlt.hpp` undoes a change that is not ORB-SLAM3's at all,
and so appears in no recorded delta: upstream switched its sparse solver from
`SimplicialLDLT` to `SimplicialLLT` in 2020. LDLT fails only on a pivot that is
exactly zero, LLT on any that is not positive -- and a visual-inertial bundle
adjustment, with its gauge free and lambda starting at 1e-5, has a Hessian that
is positive definite only on paper (the two that were dumped factor with exactly
one pivot of -0.02 each). v1.0 solved those and moved on. With upstream's solver
every inertial run logged `Cholesky failure` -- 1 to 54 times -- and each one
wrote a 1.3 MB `debug.txt` from the Local Mapping thread, about a second, before
Levenberg-Marquardt inflated lambda and tried again. `LinearSolverEigenLDLT` is
upstream's class with the decomposition swapped; `MakeBlockSolver` uses it for
`kEigen`. On EuRoC stereo-inertial (MH01, V201, V203, two runs each, the three
builds side by side): v1.0 0 failures in 6 runs, upstream's LLT 1-10 in every
one, the wrapper 0 in 6; ATE the same within its spread in all three.

### Where the remaster differs from v1.0 on purpose

Upstream fixed `Sim3`'s `exp` and `log` in 2017: for a rotation below 1e-5 rad
with a scale change above 1e-5, v1.0's fork computes the coefficient `B` of
`W = A*Omega + B*Omega^2 + C*I` without its `- 1`, and the small-angle rotation
without its `/2`. That branch is what a monocular essential-graph optimisation
sits in once scale drift is being spread along the graph. The remaster uses
upstream's, which is the correct one, and does not port the bug back.

### `dbow2_build/` — a build definition, not a code change

Upstream DBoW2's own `CMakeLists.txt` unconditionally compiles `FBrief.cpp`,
which needs DLib's `DVision`, whose `.cpp` files use the OpenCV 1.x C API
(`IplImage`, `CvScalar`, `CV_RGB2GRAY`) that OpenCV 4 removed. ORB-SLAM3 never
uses BRIEF descriptors.

Patching upstream would break the pristine guarantee, so this directory owns an
alternative build definition that compiles exactly the translation units needed,
straight from untouched upstream sources. `FORB.cpp`'s
`#include <DVision/DVision.h>` is vestigial — zero `DVision::` references — and
DLib's *headers* do compile under OpenCV 4, so the include path alone satisfies
it with no library to link.

## Verifying all of this

```bash
# thirdparty/ is still untouched upstream, on its pinned tags
./tools/check_pristine.sh

# which files ORB-SLAM3 edited in its own vendored copies
./tools/classify_vendored.sh all

# regenerate the per-file deltas
./tools/upstream_delta.py --project g2o --out docs/modifications

# the wrappers behave like the code they replace
./docker/run.sh -- bash -c '
  cmake -S /workspace/vendor_ext -B /workspace/build/vendor_ext -G Ninja &&
  cmake --build /workspace/build/vendor_ext -j 8 &&
  /workspace/build/vendor_ext/tests/test_vendor_ext /workspace/Vocabulary/ORBvoc.txt'
```

The last one loads the real 971,814-word `ORBvoc.txt` in about 2 seconds and
checks that a save/load round trip scores identically to the in-memory original.
