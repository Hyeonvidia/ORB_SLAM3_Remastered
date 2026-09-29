# Measurements

What was measured, how, and what it showed, in the order it was done: the
vendored libraries against upstream, the dataset matrix and how reproducible
it is, and the comparisons against ORB-SLAM3 v1.0. The README has the current
figures; this is the account behind them.

## What the measurement found

`tools/classify_vendored.sh` and `tools/upstream_delta.py` compare each vendored
file against upstream *history* rather than upstream's current tip, so fork drift
and reformatting are not mistaken for ORB-SLAM3's work.

| Library | Files changed | Lines |
|---|---:|---:|
| Sophus | **0 / 21** | **0** |
| DBoW2 | 7 / 14 | 237 |
| g2o | 63 / 95 | 1105 |

Findings worth knowing:

- **Sophus was never modified.** Upstream's `Dependencies.md` says it was.
- **g2o's only unique file, `types/se3mat.*`, is dead code** — nothing
  references `SE3mat`, and it duplicates `G2oTypes.h`.
- **`PnPsolver` does not exist in v1.0.** MLPnP replaced it; the doc entry stayed.
- **The whole system leaned on a leaked `using namespace std`** injected by the
  DBoW2 fork. ~2,500 names are now qualified explicitly.
- **`LoopClosing::mnFullBAIdx` is incremented and compared, but declared
  `bool`**, so it saturates at `true` and the guard it feeds is permanently dead
  after the first global-BA abort.

Details in [docs/DEPENDENCIES.md](DEPENDENCIES.md) and
[docs/PORTING.md](PORTING.md).

## Status

Builds clean on GCC 13.3 / C++17: `libORB_SLAM3.so` plus twelve dataset example
binaries, zero errors and zero undefined symbols.

69 runs across EuRoC, KITTI and TUM RGB-D all complete, and are scored with the
alignment and ground-truth frame each configuration actually calls for:

| Dataset | Runs | Median ATE | Median ATE / path |
|---|---:|---:|---:|
| EuRoC (11 seq × 4 configs) | 44 | **0.045 m** | — |
| KITTI odometry 00–10 (mono + stereo) | 22 | 3.42 m | **0.228 %** |
| TUM RGB-D fr1_desk | 3 | **0.017 m** | — |

KITTI stereo alone lands between 0.03 % and 0.62 % of path length (median
0.11 %). Monocular is far worse there, as expected: no metric scale over
kilometre-long drives, with the 2.5 km highway sequence 01 the known failure at
11.6 %.

Read those as one sample, not as the system's output. ORB-SLAM3 is several
threads racing, and how many keyframes a run ends up with depends on whether
Local Mapping kept up -- so the numbers move when nothing in the code has. Two
matrices built from the same binary and run the same way differ per cell by:

| Configuration | Cells | Median ΔATE | Worst cell |
|---|---:|---:|---:|
| RGB-D | 2 | 3 % | 5 % |
| stereo | 22 | 9 % | 38 % |
| stereo-inertial | 11 | 12 % | 63 % |
| mono | 23 | 15 % | **19x** |
| mono-inertial | 11 | 22 % | **7x** |

The two worst are not noise around a value, they are different runs: KITTI 08
mono scored 2.76 m once and 55.5 m the next time, because the first lost
tracking at 253 keyframes and the second kept all 2922. V103 mono swings 8x the
same way. Those cells cannot support a claim about a code change in either
direction.

The practical consequence, learned the hard way: a single re-run of the matrix
cannot tell you whether a change regressed anything. Run it twice on the same
build first, and compare the change against that spread -- and do not compare a
JOBS=4 matrix against sequential runs, which produce 17 % more keyframes on the
same sequence and are a different measurement.

```bash
./tools/run_all.sh                                   # the whole matrix
./docker/run.sh -- python3 /workspace/tools/eval_all.py   # score it
```

[docs/PORTING.md](PORTING.md) explains the two ways this evaluation goes
quietly wrong if the frame or the alignment group is chosen carelessly.

## Against the original

Whether any of this changed how well the system tracks is a question the
matrix alone cannot answer, so v1.0 itself was built in the same container and
run on the same images -- headless, in one queue with the remaster, so both saw
the same load. `tools/baseline/build.sh` builds it and `tools/ab_kitti.sh` runs
the comparison: KITTI 00-10, mono three times and stereo twice, 110 runs.

v1.0 does not build or run here as it is. `tools/baseline/v1.0-build.patch` is
the least that makes it: C++17 for the pinned Pangolin; `mnFullBAIdx++` on a
`bool`, which C++17 rejects (the remaster fixed the same three lines); and the
`Settings` printer dereferencing a second-camera calibration that rectified
stereo never sets -- v1.0 crashes on KITTI stereo before the first image.

Median ATE over the repeats, in metres:

| Seq | mono v1.0 | mono remaster | stereo v1.0 | stereo remaster |
|---|---:|---:|---:|---:|
| 00 | 6.34 | 9.21 | 1.18 | 1.21 |
| 01 | 352 | 314 | 14.7 | 14.9 |
| 02 | 23.7 | 27.9 | 5.03 | 5.38 |
| 03 | 1.22 | 1.24 | 1.45 | 1.38 |
| 04 | 1.10 | 1.11 | 0.26 | 0.25 |
| 05 | 7.62 | 5.92 | 0.96 | 0.95 |
| 06 | 18.2 | 15.6 | 1.01 | 1.02 |
| 07 | 2.15 | 2.46 | 0.43 | 0.43 |
| 08 | 56.9 | 56.3 | 3.91 | 3.67 |
| 09 | 39.0 | 8.50 | 1.97 | 1.96 |
| 10 | 7.77 | 7.59 | 1.22 | 1.38 |

**Accuracy is the same**, within the run-to-run spread the table further up
puts on a single build, with one exception that goes the remaster's way. On 09
mono, whose loop closes in its last frames, the remaster scored 8-10 m in all
nine runs of three comparisons; v1.0 did in five, and 39-55 m in the other
four. In the first two comparisons the remaster detected the loop six times out
of six, v1.0 three. v1.0's `Shutdown()` does not wait for its threads, so the last keyframes
may never reach Loop Closing and a correction may not land before the
trajectory is written; the remaster joins them first. That is the likely
reason, not a proven one. 01 mono fails in both, as it does for everyone.

**Tracking is slightly faster**: a median 1.8 % per frame in mono (9 of 11
sequences; 15.06 to 14.81 ms on average) and 0.5 % in stereo (18.70 to 18.57).
It was 2-6 % *slower* until the cause was found -- v1.0's g2o stops
Levenberg-Marquardt once it has stalled and upstream's does not, so every pose
optimisation ran ten iterations on a problem solved in five -- and with the
rule restored upstream's leaner iteration puts the remaster ahead.
[docs/WRAPPERS.md](WRAPPERS.md) has how it was found. `-march=native`,
which v1.0's build adds everywhere, was measured in eight combinations of
library and caller and does nothing on this machine.

**The memory work cost tracking nothing**, though the same comparison, run
again after it, seemed to say otherwise: 0.0 % in mono and +0.5 % in stereo
against v1.0, where it had been -1.8 % and -0.5 %. Two builds of the same
source, put through the same queue side by side, then came out 1.5 % apart --
that is the floor of timing whole runs several at a time, and the shift is
inside it. `tools/ab_stages.sh` is the finer instrument: it builds two commits
with the stage timers on and runs them in pairs. Before and after the memory
work, six pairs on each of three sequences:

| Stage, ms | 04 stereo | 07 mono | 07 stereo |
|---|---|---|---|
| Tracking, total | 18.53 -> 18.65 | 15.38 -> 15.72 | 19.52 -> 19.52 |
| of which creating a keyframe | 0.150 -> 0.128 | 0.045 -> 0.035 | 0.098 -> 0.090 |
| Local Mapping, inserting a keyframe | 4.26 -> 3.01 | 4.59 -> 3.27 | 3.88 -> 2.50 |
| Local Mapping, creating map points | 10.10 -> 9.71 | 22.58 -> 22.53 | 7.80 -> 7.60 |

Tracking's total differs with either sign from pair to pair, and what does
differ in 07 mono is ORB extraction, which the work did not touch. The one
stage of tracking it did touch, the keyframe constructor, got faster. Inserting
a keyframe got faster by a third in all eighteen pairs: that is where a
keyframe's bag of words is computed, which no longer builds a `cv::Mat` per
descriptor and walks a vocabulary whose nodes hold their 32 bytes in place.
Accuracy, pooled over
both comparisons: the remaster's median ATE is 1.3 % above v1.0's in stereo
(higher in 7 sequences of 11) and 0.5 % below in mono (higher in 3 of 11),
which is no difference.

**Culled map points are freed.** ORB-SLAM3 never frees a MapPoint or a KeyFrame
it culls: `SetBadFlag()` erases the pointer from the map and the object stays on
the heap until the process ends, because nobody could say who might still hold
it. On KITTI 00 stereo that is 537,000 points, 418 MB, growing about 95 KB per
frame -- the largest thing in the process after a few minutes. The remaster
frees them: a culled point is handed to the atlas, every thread announces once
per loop that it holds nothing from earlier iterations except what it names,
the atlas looks in every place a pointer can still be (every keyframe's slots,
live and culled; the reference points; the replaced-by links) and frees what
nobody names two grace periods later. A point somebody still names simply
waits; nothing is ever written into a structure in its place, so the SLAM
threads compute what they computed. [docs/OWNERSHIP.md](OWNERSHIP.md) has
the design, what was rejected and why, and the order it was built in; each step
was proven on its own -- counts against the memory report, a dry run timed
against the one before it, AddressSanitizer with a mode that poisons freed
memory instead of returning it, ThreadSanitizer on a synthetic world with
variants that must fail.

With `ORBSLAM3R_MEMORY_REPORT=1`, before and after, both under the same load:

| | culled MapPoints | peak RSS |
|---|---:|---:|
| KITTI 00 stereo | 415 MB -> 5.7 MB | 2160 -> 1676 MB |
| KITTI 00 mono | 258 MB -> 8.1 MB | 3557 -> 3087 MB |
| EuRoC MH01 stereo-inertial | 93 MB -> 18 MB | 515 -> 424 MB |
| EuRoC MH01 mono-inertial | 85 MB -> 11 MB | 656 -> 537 MB |

What stays is what the census said would stay: the points that only culled
keyframes' slots still name (one in five on MH01), which go when keyframes are
reclaimed, and the few thousand in flight when the run ends. Freeing cost
tracking at first -- new points were allocated where old ones had been freed,
scattered across the heap, and each point's descriptor was a heap block of its
own whose freeing scattered everything allocated after it. MapPoints now come
from a pool of their own (`atlas/SlotPool`) and carry their descriptor inside;
with that, freeing against not freeing is -0.3 ms of tracking per frame on
KITTI stereo, the freeing side faster. Against v1.0, the full KITTI comparison, run again with everything on (110 runs),
puts tracking at -0.6 % per frame in stereo, faster in 9 sequences of 11,
and 0.0 % in mono, with either sign; accuracy is the same within the
spread, and 09 mono closed its loop in all three runs against v1.0's two.
Before the pool and the inline descriptor the same comparison had said
+3.3 % in stereo, which is why both exist.
`ORBSLAM3R_RECLAIM=dry` turns freeing off and leaves the rest running, for a
comparison on any sequence.

`ORBmatcher::SearchForTriangulation` let several features of one keyframe take
the same feature of another (v1.0 tests `vbMatched2` and never sets it), so
points were made twice and one of each pair observed a keyframe that did not
know it. With that fixed the comparison was stopped at 78 runs of 110
(sequences 00-07): tracking -2.2 % per frame in stereo, faster in all eight,
-0.4 % in mono with either sign, accuracy the same within the spread.

**How much to run.** The 110 runs take two hours and are for a release. For a
change, `./tools/ab_kitti.sh 04 07` (20 runs, six minutes) or
`./tools/ab_stages.sh <before> <after>`, which times the stages the change
touches and uses the ones it does not as its control. v1.0 is a reference
here, not a limit: what a change is held to is the remaster's own last
figures.

**Inertial configurations** had a second difference, found by auditing every
change between the two g2o's rather than by measuring: upstream's sparse solver
refuses the numerically indefinite Hessians an inertial BA produces, and v1.0's
does not. EuRoC stereo-inertial, MH01 / V201 / V203, two runs of each build
side by side:

| Build | Factorisation failures per run | ATE, m |
|---|---|---|
| v1.0 | 0 0 0 0 0 0 | 0.045 0.034 / 0.033 0.030 / 0.024 0.042 |
| remaster, upstream's solver | 1 10 1 1 1 4 | 0.042 0.036 / 0.035 0.034 / 0.042 0.024 |
| remaster, as it is now | 0 0 0 0 0 0 | 0.037 0.036 / 0.036 0.036 / 0.048 0.022 |

Each failure cost Local Mapping about a second. With six runs the ATE does not
separate the three, but under the heavier load of the full matrix the same
failures came twenty in a row and tracking was lost inside them.

An A/B is only as good as the machine's attention: the laptop slept through 30
of the 110 runs the first time, which left their trajectories intact and their
timings useless -- a median tracking time of 28 ms where 18 is normal. Those
30 were run again. Wall time far beyond a sequence's length is the symptom.

