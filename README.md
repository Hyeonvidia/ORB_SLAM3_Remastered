# ORB_SLAM3_Remastered

ORB-SLAM3 rebuilt for a robot's onboard computer: monocular, stereo and RGB-D
SLAM that tracks a frame in a little over half the time of ORB-SLAM3 v1.0 and
runs in half the memory or less, at the same accuracy, with the defects found
on the way fixed and the code laid out along the architecture of the paper.

![KITTI 07, stereo: the frame with its features, and the map being built](docs/media/kitti07_stereo.gif)

## At a glance

![Tracking time per frame and peak memory, v1.0 against this repository](docs/media/at_a_glance.png)

| Run | Tracking per frame, ms | Peak memory, MB | ATE, m |
|---|---:|---:|---:|
| KITTI 07 monocular | 13.3 → **7.9** (-41 %) | 1545 → **1021** (-34 %) | 2.13 → 2.79 |
| KITTI 07 stereo | 17.1 → **9.5** (-44 %) | 896 → **449** (-50 %) | 0.408 → 0.402 |
| TUM fr1_desk RGB-D | 12.5 → **8.3** (-33 %) | 808 → **304** (-62 %) | 0.018 → 0.018 |
| EuRoC V101 stereo | – → 7.3 | 752 → **274** (-64 %) | 0.037 → 0.039 |

v1.0 → now, medians of three runs of each, four runs sharing the machine
(linux/arm64 in Docker, Apple M-series). Monocular accuracy on KITTI moves by
more than this from one run to the next of the same binary.

**Place recognition** finds what is there to be found:

| Run | Loops closed | Maps merged | Tracking per frame, ms | ATE, m |
|---|---:|---:|---:|---:|
| KITTI 05 stereo (three loops) | 3 | – | 10.7 | 0.94 |
| KITTI 05 monocular | 3 | – | 8.1 | 4.76 |
| EuRoC V101 + V102 in one session, stereo | 2 | 1 | 7.6 | 0.037 |
| EuRoC V101 + V102 in one session, monocular | 1 | 1 | 7.7 | 0.029 |

Both tables come from one command, which takes three minutes (five for the
second) and compares with what was last accepted:

```bash
./tools/check.sh            # speed, accuracy, memory: mono, stereo, RGB-D
./tools/check.sh robust     # loops and merges
```

## What was done

**Faster**

| | Before | After |
|---|---:|---:|
| ORB extraction, KITTI image, 2000 features | 9.8 ms | 4.5 ms on two threads, 3.3 on three |
| – of which descriptors | 1.9 ms | 0.3 ms |
| Tracking on KITTI 07 stereo after that, with stereo matching and descriptor comparison reworked | 11.5 ms | 9.5 ms |

- The levels of the image pyramid are extracted in parallel; what is extracted
  does not depend on the number of threads (`ORBextractor.nThreads`).
- A descriptor's 512 sampling points are rotated four at a time (SSE2 / NEON).
- Descriptors are compared 64 bits at a time, inline, in every search.
- Stereo matching sums its windows on the pixels instead of making a matrix
  for each.

**Smaller**

- MapPoints that were culled are freed once no thread can still be using them
  ([docs/OWNERSHIP.md](docs/OWNERSHIP.md)); v1.0 never frees one.
- A KeyFrame is a third smaller: flat feature grid, no duplicate keypoints.
- The vocabulary holds its descriptors as 32 bytes each, not as a matrix each.

**Defects of v1.0 that were fixed**

| Defect | Effect |
|---|---|
| `Sim3Solver` divided 0 by 0 for an identity rotation | the process aborted without a message, in about 3 % of runs |
| `DetectNBestCandidates` did not advance past a culled candidate | Loop Closing hung |
| `SearchForTriangulation` never marked a feature as taken | duplicate map points, observations a keyframe did not know of |
| The projection Jacobian kept a reference to a temporary | undefined behaviour in every pose optimisation |
| Two Frame copies leaked per frame after an inertial relocalisation | memory grew without bound |
| `UpdateLocalKeyFrames` iterated a vector it was growing | undefined behaviour |
| Threads were not joined at shutdown; trajectory savers read an uninitialised map | crashes on exit |
| `mnFullBAIdx` was declared `bool` and counted | the guard it feeds was dead after the first global BA |

**Maintainable**

- Sources in layers that follow Figure 1 of the paper: `atlas/`, `tracking/`,
  `local_mapping/`, `loop_closing/`, `optimization/`, and below them `camera/`,
  `features/`, `common/`. No include cycles; `tools/cycles.py` keeps it so.
- Dependencies are pinned upstream releases, untouched, in `thirdparty/`;
  ORB-SLAM3's changes to them are separate code in `vendor_ext/`
  ([docs/WRAPPERS.md](docs/WRAPPERS.md)).
- Built and run in Docker only. `std::` written out, no `using namespace`.
- Tests in seven seconds (`ctest`), a benchmark of the extractor on its own
  (`tests/bench_orb`), and the two checks above.

## Quick start

```bash
git clone --recurse-submodules <this repo> ORB_SLAM3_Remastered
cd ORB_SLAM3_Remastered
./docker/build_images.sh                      # base -> thirdparty -> dev, ~10 min
tar -xzf reference/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
./tools/build.sh                              # library, examples, tests
./tools/check.sh                              # needs the datasets under /datasets
./tools/monitor.sh kitti stereo 07            # watch a sequence, from macOS
```

Datasets are expected where `tools/dataset_plan.sh` says: EuRoC, KITTI
odometry and TUM RGB-D.

## Documentation

| | |
|---|---|
| [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) | what was measured and how: the dataset matrix, its reproducibility, every comparison against v1.0 |
| [docs/OWNERSHIP.md](docs/OWNERSHIP.md) | who owns a KeyFrame or a MapPoint, and how a culled one is freed |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | the Docker images, building, formatting, layout of the tree |
| [docs/VIEWER.md](docs/VIEWER.md) | watching the viewer live from macOS |
| [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md), [docs/WRAPPERS.md](docs/WRAPPERS.md) | the borrowed libraries, and ORB-SLAM3's changes to them |
| [docs/PORTING.md](docs/PORTING.md), [docs/FRAME_KEYFRAME.md](docs/FRAME_KEYFRAME.md) | how the port was made; why Frame and KeyFrame share what they share |

## Known and open

- On TUM fr1_desk RGB-D, tracking is lost right after initialisation in about
  one run in ten and recovered by a second map that is merged later: v1.0 in 2
  runs of 12, this tree in 1 of 12.
- FAST detection is what is left of extraction's cost, 5.5 ms of 8 on one
  thread.
- The back end -- local bundle adjustment, point creation, keyframe culling --
  is as v1.0 left it.
- Culled KeyFrames are not freed yet; cross-thread reads of public members and
  the cycle between the three threads' classes remain.

## License

ORB-SLAM3 is GPLv3; see [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for the
license of every borrowed component.
