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

## The matrix at d08b4c6 (2026-10-03)

Every configuration once, four runs sharing the machine, after the work of
September 30 to October 2 (parallel extraction, one-pass FAST, flat grids,
observations and words, the compact vocabulary, the local bundle adjustment's
damping in stereo and RGB-D, the thread ports). 69 runs, none failed.

| Dataset | Runs | Median ATE | Median ATE / path |
|---|---:|---:|---:|
| EuRoC (11 seq x 4 configs) | 44 | **0.039 m** | 0.051 % |
| KITTI odometry 00-10 (mono + stereo) | 22 | 3.93 m | 0.230 % |
| TUM RGB-D fr1_desk | 3 | **0.017 m** | 0.184 % |

Against the matrix of September 19, cell by cell, 23 cells moved by more than
20 % either way: 13 better (MH05 stereo-inertial 0.387 -> 0.050, V201 stereo
0.049 -> 0.030, TUM mono 0.035 -> 0.014 among them) and 10 worse, five of
them stereo by 1.6 to 2.4 times: MH03 (0.026 -> 0.045), MH04 (0.043 ->
0.072), V202 (0.042 -> 0.073), V203 (0.378 -> 0.900), KITTI 01 (15.1 ->
27.6).

Those five were run twice more with this build, with the local bundle
adjustment change (421b038) reverted, and with the one-pass FAST (9affa4d)
reverted -- the two changes that touch stereo and not monocular:

| ATE, m, two runs | this build | without 421b038 | without 9affa4d |
|---|---:|---:|---:|
| MH03 stereo | 0.024 / 0.033 | 0.035 / 0.025 | 0.048 / 0.042 |
| MH04 stereo | 0.069 / 0.057 | 0.098 / 0.088 | 0.080 / 0.085 |
| V202 stereo | 0.056 / 0.096 | 0.043 / 0.057 | 0.072 / 0.076 |
| V203 stereo | 0.465 / 0.313 | 0.804 / 0.288 | 0.583 / 0.863 |
| KITTI 01 stereo | 20.3 / 12.8 | 15.8 / 14.8 | 16.6 / 13.2 |

The ranges overlap in every row; V203 and KITTI 01 end anywhere between 0.29
and 0.90 m and 12.8 and 27.6 m with any of the three. Neither change is what
the matrix's worse cells came from; they are what one run of those sequences
looks like next to another. The full scores:

```
EuRoC  (11 sequences x 4 configurations)
RUN                       ATE RMSE      PATH  ATE/PATH   PAIRS  NOTE
----------------------    --------  --------  --------   -----  ----------------------
MH01 mono                   0.0180     80.6m    0.022%    3638  Sim(3)
MH01 stereo                 0.0403     80.6m    0.050%    3638  SE(3), scale err +0.76%
MH01 mono_inertial          0.0770     71.2m    0.108%    2644  SE(3), scale err +1.84%
MH01 stereo_inertial        0.0339     80.5m    0.042%    3639  SE(3), scale err +0.58%
MH02 mono                   0.0151     73.4m    0.021%    2998  Sim(3)
MH02 stereo                 0.0244     73.4m    0.033%    2999  SE(3), scale err -0.09%
MH02 mono_inertial          0.0603     62.0m    0.097%    2142  SE(3), scale err -1.04%
MH02 stereo_inertial        0.0373     73.4m    0.051%    3000  SE(3), scale err -0.26%
MH03 mono                   0.0284    130.8m    0.022%    2623  Sim(3)
MH03 stereo                 0.0447    130.9m    0.034%    2631  SE(3), scale err -0.77%
MH03 mono_inertial          0.0356    124.3m    0.029%    2147  SE(3), scale err -0.11%
MH03 stereo_inertial        0.0352    130.7m    0.027%    2631  SE(3), scale err +0.38%
MH04 mono                   0.0515     91.7m    0.056%    1960  Sim(3)
MH04 stereo                 0.0716     91.7m    0.078%    1976  SE(3), scale err +0.69%
MH04 mono_inertial          0.1501     91.4m    0.164%    1952  SE(3), scale err -2.05%
MH04 stereo_inertial        0.0453     91.6m    0.049%    1976  SE(3), scale err +0.29%
MH05 mono                   0.0656     97.4m    0.067%    2164  Sim(3)
MH05 stereo                 0.0423     97.5m    0.043%    2221  SE(3), scale err -0.28%
MH05 mono_inertial          0.1777     89.6m    0.198%    1697  SE(3), scale err -2.21%
MH05 stereo_inertial        0.0500     97.5m    0.051%    2170  SE(3), scale err -0.26%
V101 mono                   0.0335     58.4m    0.057%    2781  Sim(3)
V101 stereo                 0.0357     58.5m    0.061%    2871  SE(3), scale err +0.46%
V101 mono_inertial          0.0437     58.3m    0.075%    2769  SE(3), scale err +1.58%
V101 stereo_inertial        0.0386     58.5m    0.066%    2790  SE(3), scale err +1.01%
V102 mono                   0.0115     75.5m    0.015%    1595  Sim(3)
V102 stereo                 0.0514     75.5m    0.068%    1671  SE(3), scale err +0.65%
V102 mono_inertial          0.0215     73.9m    0.029%    1519  SE(3), scale err +0.83%
V102 stereo_inertial        0.0218     75.8m    0.029%    1601  SE(3), scale err +0.73%
V103 mono                   0.3906     79.3m    0.492%    1995  Sim(3)
V103 stereo                 0.1728     79.3m    0.218%    2094  SE(3), scale err +0.67%
V103 mono_inertial          0.0370     74.1m    0.050%    1879  SE(3), scale err +2.13%
V103 stereo_inertial        0.0239     78.9m    0.030%    1994  SE(3), scale err +1.03%
V201 mono                   0.0180     36.2m    0.050%    2075  Sim(3)
V201 stereo                 0.0297     36.3m    0.082%    2147  SE(3), scale err +0.55%
V201 mono_inertial          0.0390     36.4m    0.107%    2167  SE(3), scale err -0.10%
V201 stereo_inertial        0.0356     36.4m    0.098%    2179  SE(3), scale err +1.15%
V202 mono                   0.0160     83.6m    0.019%    2269  Sim(3)
V202 stereo                 0.0728     83.6m    0.087%    2309  SE(3), scale err +0.01%
V202 mono_inertial          0.0156     82.9m    0.019%    2249  SE(3), scale err -0.01%
V202 stereo_inertial        0.0124     83.1m    0.015%    2262  SE(3), scale err -0.18%
V203 mono                   0.0761     84.1m    0.090%    1709  Sim(3)
V203 stereo                 0.9005     79.8m    1.129%    1669  SE(3), scale err -11.79%
V203 mono_inertial          0.0281     85.9m    0.033%    1796  SE(3), scale err -0.34%
V203 stereo_inertial        0.0589     86.0m    0.069%    1805  SE(3), scale err +0.14%
                                                      
median of 44               0.0386m              0.051%

KITTI odometry  (sequences 00-10, the ones with ground truth)
RUN                       ATE RMSE      PATH  ATE/PATH   PAIRS  NOTE
----------------------    --------  --------  --------   -----  ----------------------
00 mono                     6.0458   3717.6m    0.163%    2845  Sim(3), keyframes
00 stereo                   1.1095   3724.2m    0.030%    4541  SE(3), per frame
01 mono                   330.9646   2453.2m   13.491%     661  Sim(3), keyframes
01 stereo                  27.5736   2453.2m    1.124%    1101  SE(3), per frame
02 mono                    28.7383   5067.1m    0.567%    3628  Sim(3), keyframes
02 stereo                   4.7397   5067.2m    0.094%    4661  SE(3), per frame
03 mono                     0.9522    560.8m    0.170%     499  Sim(3), keyframes
03 stereo                   1.6333    560.9m    0.291%     801  SE(3), per frame
04 mono                     1.1122    393.6m    0.283%     195  Sim(3), keyframes
04 stereo                   0.2821    393.6m    0.072%     271  SE(3), per frame
05 mono                     5.0695   2204.3m    0.230%    1716  Sim(3), keyframes
05 stereo                   1.0296   2205.6m    0.047%    2761  SE(3), per frame
06 mono                    14.7245   1232.8m    1.194%     747  Sim(3), keyframes
06 stereo                   1.1370   1232.9m    0.092%    1101  SE(3), per frame
07 mono                     3.9331    694.3m    0.566%     707  Sim(3), keyframes
07 stereo                   0.4279    694.7m    0.062%    1101  SE(3), per frame
08 mono                    60.8801   3222.4m    1.889%    3040  Sim(3), keyframes
08 stereo                   3.6682   3222.8m    0.114%    4071  SE(3), per frame
09 mono                     9.4515   1705.0m    0.554%    1258  Sim(3), keyframes
09 stereo                   1.9948   1705.1m    0.117%    1591  SE(3), per frame
10 mono                     7.1821    919.4m    0.781%     934  Sim(3), keyframes
10 stereo                   1.0027    919.5m    0.109%    1201  SE(3), per frame
                                                      
median of 22               3.9331m              0.230%

TUM RGB-D  (freiburg1_desk)
RUN                       ATE RMSE      PATH  ATE/PATH   PAIRS  NOTE
----------------------    --------  --------  --------   -----  ----------------------
fr1_desk mono               0.0140      9.3m    0.152%     154  Sim(3), keyframes
fr1_desk rgbd               0.0171      9.3m    0.184%     573  SE(3), per frame
fr1_desk rgbd (kf)          0.0196      9.2m    0.214%     152  SE(3), keyframes
                                                      
median of 3                0.0171m              0.184%
```
