# Watching the viewer

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

While a run is going, the log under the map reports progress every five
seconds — frame, frame rate, keyframes, map points — and every change of
tracking state; the system itself prints nothing between "New Map created" and
"Shutdown", and a log that never moves reads as a hang. When the last run ends
the window stays up with the final map — the status row turns amber and says
FINISHED — until Esc or the Stop button in the window ends it. `--hold-each`
pauses that way after every run; `--no-hold` closes the window with the last
frame, as upstream does, which on KITTI 04 (27 seconds of driving) looks like a
crash. On the binary itself it is `ORBSLAM3R_VIEWER_HOLD=1`.

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


