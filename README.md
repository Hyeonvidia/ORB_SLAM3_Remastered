# ORB_SLAM3_Remastered

ORB-SLAM3 v1.0 rebuilt on **current** upstream dependencies, developed and run
in Docker, with every change to a borrowed library visible as separate code
rather than buried in a fork.

Three things make it different from the upstream repository:

1. **Dependencies are pinned git submodules, checked out untouched.**
   `thirdparty/` holds byte-identical upstream releases — not the stripped,
   edited copies ORB-SLAM3 ships.
2. **ORB-SLAM3's changes to those libraries live in `vendor_ext/`** as wrappers,
   subclasses and aliases you can read on their own. See
   [docs/WRAPPERS.md](docs/WRAPPERS.md).
3. **Nothing is built on the host.** A three-layer image stack gives a
   reproducible linux/arm64 toolchain; the macOS side only edits files.
4. **The port from the pristine reference is scripted and replayable**, so the
   2,500 `std::` qualifications and the bug fixes are auditable rather than
   asserted. The scripts have since been frozen and the tree is maintained by
   hand; see [docs/PORTING.md](docs/PORTING.md).
5. **The sources are laid out along the architecture in the paper** — an Atlas
   layer, one folder per thread, and the camera abstraction of Section IV — so
   an include line says which layer it reaches into.

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
tools/            submodule pinning, the port pipeline, ATE evaluation, format.sh
docs/             DEPENDENCIES.md, WRAPPERS.md, PORTING.md, FRAME_KEYFRAME.md,
                  modifications/
```

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

Details in [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) and
[docs/PORTING.md](docs/PORTING.md).

## Status

Builds clean on GCC 13.3 / C++17: `libORB_SLAM3.so` plus twelve dataset example
binaries, zero errors and zero undefined symbols.

69 runs across EuRoC, KITTI and TUM RGB-D all complete, and are scored with the
alignment and ground-truth frame each configuration actually calls for:

| Dataset | Runs | Median ATE | Median ATE / path |
|---|---:|---:|---:|
| EuRoC (11 seq × 4 configs) | 44 | **0.043 m** | — |
| KITTI odometry 00–10 (mono + stereo) | 22 | 3.09 m | **0.204 %** |
| TUM RGB-D fr1_desk | 3 | **0.017 m** | — |

KITTI stereo alone lands between 0.03 % and 0.59 % of path length (median
0.09 %). Monocular is far worse there, as expected: no metric scale over
kilometre-long drives, with the 2.5 km highway sequence 01 the known failure at
11.6 %.

```bash
./tools/run_all.sh                                   # the whole matrix
./docker/run.sh -- python3 /workspace/tools/eval_all.py   # score it
```

[docs/PORTING.md](docs/PORTING.md) explains the two ways this evaluation goes
quietly wrong if the frame or the alignment group is chosen carelessly.

## Watching the viewer live from macOS

```bash
./tools/run_gui.sh euroc                     # every EuRoC sequence, all 4 configurations
./tools/run_gui.sh euroc stereo              # one configuration
./tools/run_gui.sh kitti stereo 04 05        # named sequences
./tools/run_gui.sh --list all                # what would run, without running it
./tools/run_gui.sh --vnc tum                 # over Screen Sharing instead

./tools/monitor.sh euroc stereo-inertial V203   # one sequence; a thin alias for the above
```

`run_gui.sh` walks a whole dataset with the viewer on, one sequence after
another, under a single nested X server — the window opens once and stays up for
the series rather than flickering in and out between runs. Trajectories land in
`results/gui/<tag>/`, one directory per run, which is what lets
`tools/evaluate_ate.py` score them afterwards.

These are slow: rendering is software llvmpipe inside the container, so a full
EuRoC sweep is 44 runs and takes hours. `--list` first.

Which binary and which arguments each (dataset, configuration, sequence) needs
lives in one place, `tools/dataset_plan.sh`, shared with the headless
`tools/run_all.sh`.

A real window opens on the macOS desktop through XQuartz, and the mouse
works — toggling a menu checkbox moves the map-point pixel count between 638, 0
and 970, and a drag orbits the view. There is no window manager inside the
nested server, so input focus is PointerRoot: whatever is under the pointer gets
both the clicks and the keys. If a click seems to do nothing, click once on the
window to activate it first; macOS swallows the activating click unless
*Click-through inactive windows* is enabled in XQuartz's settings.

The window is three rows to the right of the menu: the tracked frame on top, the
3D map in the middle, the system's own log messages at the bottom. Side by side
was worse, and a wide sensor shows why — a KITTI frame is 1226x370, so a column
wide enough to show it left two thirds of that column empty underneath while the
map was squeezed into what remained. Stacking gives the map the full window
width, which is also the shape a driving trajectory wants.

The frame takes the height its own aspect needs, capped so it cannot crowd out
the map; `ORBSLAM3R_FRAME_VIEW_FRACTION` pins it instead, without a rebuild:

```bash
ORBSLAM3R_FRAME_VIEW_FRACTION=0.5 ./tools/run_gui.sh euroc stereo MH01
```

Between the frame and the map is a status row: the sensors the run was started
with, then what the system is doing and the map's counts, all on one line.

```
STEREO-INERTIAL  |  SLAM MODE  |  Maps: 1, KFs: 127, MPs: 15169, Matches: 366
```

The sensor configuration is there because the viewer is often the only thing
being watched, and neither the picture nor the trajectory says whether the IMU
was in the loop.

The row is drawn by the viewer at window resolution rather than into the image.
Written into the image the line is part of the texture and shrinks with the
frame, so `cv::putText` at single-pixel strokes came apart the moment the frame
was shown under 1:1 — and the black band it needed changed the frame's aspect,
which fed back into the layout above.

### Why it goes through a nested X server

Pointing the viewer straight at XQuartz gives a window that appears with the
right title and size and then stays blank. Rendering is not the problem — a
probe reports a correct viewport, frames drawn and no GL error — presentation
is. Mesa logs `Failed to attach to x11 shm` exactly once per frame and the
window never fills in. MIT-SHM cannot work between the container's Linux VM and
macOS, since there is no shared segment across two kernels, and Mesa 25's
software X11 path has no working fallback; `LIBGL_KOPPER_DISABLE`,
`LIBGL_DRI3_DISABLE`, `GALLIUM_DRIVER=softpipe` and `LIBGL_ALWAYS_INDIRECT`
change nothing.

Xephyr breaks the chain in the right place. It is an X server that runs inside
the container and owns a framebuffer there, so Mesa presents to it locally with
shared memory working normally. Xephyr then repaints its own window on XQuartz —
and unlike Mesa it handles the missing extension, logging `Xephyr unable to use
SHM XImages` once and falling back to plain `XPutImage`.

`--vnc` remains for machines without XQuartz: the container renders to its own
Xvfb and x11vnc exports it to Screen Sharing. The password is `orbslam3r`
(`ORBSLAM3R_VNC_PASSWORD` overrides) and it is not decoration — macOS Screen
Sharing never finishes the handshake against a server offering only RFB security
type 1 (None). The port is published to 127.0.0.1 only.

Either way Mesa rasterises **inside the container** with llvmpipe; the host GPU
is not involved. XQuartz's own GLX advertises only OpenGL 1.4 with no direct
rendering, which would not be enough for Pangolin — but it never has to be,
because only finished images cross to the X server.

Not ported: the RealSense examples (need librealsense2, unavailable for
linux/arm64 here and with no camera reachable from a container) and the ROS
nodes (target ROS Melodic on Ubuntu 18.04).


## License

ORB-SLAM3 is GPLv3; see [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for the
license of every borrowed component.
