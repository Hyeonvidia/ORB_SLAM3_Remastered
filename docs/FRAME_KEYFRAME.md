# Frame and KeyFrame: what they share, and how the sharing is expressed

`Frame` and `KeyFrame` overlap heavily -- enough that the obvious move is to
make one inherit from the other. This is the measurement that says not to, and
what was done instead.

## What is actually shared

Measured on this tree:

| | `Frame` | `KeyFrame` |
|---|---:|---:|
| methods | 41 | 74 |
| member variables | 66 | 118 |
| mutexes | 1 | 4 |

16 methods share a name, 12 of them with identical signatures: `ComputeBoW`,
`GetCameraCenter`, `GetImuPose`, `GetImuPosition`, `GetImuRotation`, `GetPose`,
`GetRelativePoseTlr`, `GetRelativePoseTrl`, `GetVelocity`,
`PrintPointDistribution`, `ProjectPointDistort`, `SetNewBias`. Four more share a
name but not a signature: `GetFeaturesInArea`, `SetPose`, `SetVelocity`,
`UnprojectStereo`. 37 member variables are common to both.

The sharpest evidence is the constructor: `KeyFrame::KeyFrame(Frame&, Map*,
KeyFrameDatabase*)` copies **38 fields out of the Frame by hand** in its
initialiser list.

## Why not inheritance

`KeyFrame : Frame` fails on the thread contract, which is the 1-versus-4 mutex
column above.

A `Frame` is a transient value. Tracking makes one per image, copies it freely
(`mLastFrame = mCurrentFrame`), and no other thread ever sees it, so its
accessors take no lock. A `KeyFrame` is heap-allocated, lives in the map, and is
read and written concurrently by Tracking, Local Mapping, Loop Closing and the
viewer; every accessor takes `mMutexPose`, `mMutexConnections`,
`mMutexFeatures` or `mMutexMap` first.

Inheriting one from the other would hand the derived class a set of unguarded
accessors it must remember to override -- and `Frame`'s accessors are `inline`
and non-virtual, so a `Frame&` to a `KeyFrame` would silently read shared state
without a lock. It would also make slicing a `KeyFrame` into a `Frame` compile.

They are also different *kinds* of thing in the paper's terms: `KeyFrame` is a
node of the Atlas -- it carries the covisibility graph, the spanning tree, loop
and merge edges, a `Map*` and a bad flag -- while `Frame` is an input to
Tracking. The 37 shared variables are not an is-a relationship; they are the
same *measurements*, held by two objects with different lifetimes.

## What was done: `NavState`

The pose and velocity block was the half of the overlap that was fully
encapsulated -- `private` in `Frame`, `protected` in `KeyFrame`, and referenced
**zero** times from anywhere else in the tree. It is now one value type,
`common/NavState.hpp`, composed into both.

Each class kept every public accessor it had, with the same signature and the
same locking, so no call site changed. What changed is underneath:

- `Frame` stored `mTcw, mRcw, mRwc, mOw, mtcw` and refreshed the derived four by
  hand in `UpdatePoseMatrices()`. `KeyFrame` stored `mTcw, mRcw, mRwc, mTwc,
  mOwb` and refreshed them inline in `SetPose`. Both are now derived in one
  place, `NavState::SetPose`, and cannot go out of step.
- `UpdatePoseMatrices()` is gone, along with its one caller. It existed only to
  re-derive values that are now never stale.
- `Frame::mbIsSet` is gone. It was a second copy of `mbHasPose`: both were
  written in exactly the same two places and never held different values.
  `isSet()` still exists and now reports `HasPose()`.
- `Frame::GetImuPose()` recomputed `mTcw.inverse()` on every call although the
  inverse was already cached. It reads the cache now.

### A defect this removed

`KeyFrame::SetPose` updated `mOwb` **only** when an IMU calibration was set:

```cpp
if (mImuCalib.mbIsSet)
    mOwb = mRwc * mImuCalib.mTcb.translation() + mTwc.translation();
```

`mOwb` is not initialised by any `KeyFrame` constructor, so on a pure-visual
keyframe `GetImuPosition()` returned whatever the allocation happened to hold.
Nothing reaches it today -- every live caller is on an inertial path, where the
calibration is set -- so this was latent rather than a live fault. `NavState`
computes it unconditionally; with no calibration `mTcb` is the identity and the
result is the camera centre, which is a defined answer rather than an
uninitialised read.

`Frame` never had the bug: it computed the IMU position at read time instead of
caching it.

## What was deliberately left alone

The other half of the overlap is the feature data -- keypoints, descriptors,
stereo matches, the grid, the scale pyramid tables, the image bounds and the
calibration. Extracting it into a shared `ImageFeatures` type is the natural
next step and it was **not** taken, because the measurement says it is a much
larger change than the pose half:

- **212 external references.** Unlike the pose fields, these members are public
  and read directly all over the tree (`mvKeysUn` alone: 64 sites, across
  `ORBmatcher`, `Optimizer`, `Tracking`, `LocalMapping`, `LoopClosing`,
  `Sim3Solver`, `MLPnPsolver`, `FrameDrawer`). Every one becomes
  `pKF->mFeatures.mvKeysUn[i]` -- longer, and no safer, since the members stay
  public either way.
- **The two are not the same type.** `Frame`'s calibration (`fx, fy, cx, cy,
  invfx, invfy`, the grid element sizes, the image bounds) is `static` -- global
  mutable state shared by every Frame -- while `KeyFrame`'s is `const` and
  per-instance. The grids differ too: `Frame` uses a C array
  `std::vector<size_t> mGrid[64][48]`, `KeyFrame` a
  `vector<vector<vector<size_t>>>` because Boost has to serialise it.

The prize is real -- a `shared_ptr<const ImageFeatures>` would collapse the
38-field copy and stop every keyframe duplicating its parent frame's keypoints
and descriptors -- but it needs `Frame`'s static calibration retired first, and
that is its own change.
