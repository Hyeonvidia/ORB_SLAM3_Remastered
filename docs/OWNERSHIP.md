# Who owns a KeyFrame or a MapPoint, and when a culled one is freed

ORB-SLAM3 never frees a MapPoint or a KeyFrame it culls. `SetBadFlag()` and
`Replace()` erase the pointer from the map's set -- `Map::EraseMapPoint` says
so itself, "This only erase the pointer" -- and the object stays on the heap
until the process ends, because nobody can say who might still hold it. That
is the largest single thing in the process after a few minutes:

| run | culled MapPoints | culled KeyFrames | peak RSS |
|---|---:|---:|---:|
| KITTI 00 stereo | 537,076 = 418 MB, growing about 95 KB a frame | 0 | 2165 MB |
| KITTI 00 mono | 347,817 = 271 MB | 15 = 5 MB | 3670 MB |
| EuRoC MH01 stereo-inertial | 119,482 = 93 MB | 474 = 97 MB | 514 MB |

This note is the design for ending that. It was reached by mapping every
holder of a `MapPoint*` and a `KeyFrame*`, designing the reclamation three
ways, and then having four reviewers try to break the design that was chosen.
They did break it, in the places listed under "What was rejected"; what is
described here is what was left standing, plus a measurement that settled the
argument the designs could not.

## Smart pointers are not the answer here

`std::shared_ptr` for graph objects was measured and ruled out. Copying and
reading 2000 pointers, as `GetMapPointMatches()` does every time it is called:

| | raw | `shared_ptr` |
|---|---:|---:|
| one thread | 1.55 us | 5.0 us |
| three threads on the same objects | 1.4 us | 108.7 us |

The count is a shared atomic, so every copy on every thread writes the same
cache line. The covisibility graph is also cyclic (keyframe to point to
keyframe), which reference counts do not collect. `std::unique_ptr` is right
for objects that do have one owner and no graph (solvers, the extractors); it
is not what this problem needs.

## The rule

- A live MapPoint or KeyFrame is owned by its `Map` (`mspMapPoints`,
  `mspKeyFrames`). A culled one is owned by the atlas's `Reclaimer`.
- Every other pointer, everywhere, is raw and does not own.
- `Tracking::mlpTemporalPoints` keeps owning its visual-odometry points, which
  never enter a map.
- Nothing is freed at the moment it is culled. Local BA goes on writing to
  points it has just culled (`Optimizer.cpp`, the `vToErase` loop and then
  `SetWorldPos`), so "culled" cannot mean "gone".

## What the census says

`ORBSLAM3R_MEMORY_REPORT=1` ends with a census of who still names the culled
objects once the threads have stopped (`atlas/MemoryAudit.cpp`). Each object is
counted once, under the first holder that names it.

| culled MapPoints named by | KITTI 00 stereo | KITTI 00 mono | MH01 stereo-inertial |
|---|---:|---:|---:|
| a slot of a keyframe still in a map | 0.1 % | 0.1 % | 0 |
| only slots of culled keyframes | 0 | 0.2 % | 20.1 % |
| only another point's replaced-by link | 2.1 % | 5.1 % | 2.2 % |
| no keyframe and no point | **97.8 %** | **94.6 %** | **77.7 %** |

| culled KeyFrames named by (MH01) | |
|---|---:|
| a live point's observations | 40.7 % |
| a live keyframe's covisibility list | 0.2 % |
| none of these | 59.1 % |

and 323 of those observations are by a live point that is *not* in the culled
keyframe's own slot.

So:

1. **A culled point that somebody still names can simply wait.** Stale slots in
   live keyframes are one in a thousand. There is no need to write anything
   into a slot in the point's place, which is what makes the rest simple.
2. **MapPoints first.** They are the part that grows with every frame, and on
   KITTI they are all of it.
3. **A culled keyframe's payload cannot be released on the strength of its own
   slots.** Live points go on observing culled keyframes, and
   `MapPoint::UpdateNormalAndDepth` reads, without asking `isBad()`, the pose of
   every keyframe in a point's observations and the keypoints of its reference
   keyframe. KeyFrames get their own design, after the MapPoints are done and
   with this table in hand.

## MapPoints: retire, wait, check, free

**Retire.** `MapPoint::SetBadFlag` and `Replace` end in
`Map::RetireMapPoint(this)` instead of `Map::EraseMapPoint(this)`: erase from
the set as today, then one `push_back` into the Reclaimer's incoming list under
its mutex. A state byte in the point (in existing padding) makes retiring
idempotent -- `SearchAndFuse` can `Replace` the same point twice, and a second
record for a point whose batch has already gone would be a double free.
`Map::EraseMapPoint` stays for the map-to-map moves of a merge.

**Wait.** Readers are the threads: Tracking, Local Mapping, Loop Closing, the
viewer, and a running Global BA. Each announces at a point in its loop where it
holds no pointer obtained earlier, except in members, which it publishes as
pins:

| reader | announces | pins |
|---|---|---|
| Tracking | in `GrabImage*`, before `Track()` -- so `Track()`'s object code does not change | `mLastFrame.mvpMapPoints`, `mvpLocalMapPoints`, and the keyframe members |
| Local Mapping | bottom of `Run()`, when not stopped | `mpCurrentKeyFrame`, `mlpRecentAddedMapPoints` |
| Loop Closing | bottom of `Run()` and on its `continue` path | its matched-point vectors, candidate and current keyframes, the queue |
| viewer | top of its loop; registered only while it runs | none |
| Global BA | online from before the thread starts until it ends; never announces | -- |

An announce costs one relaxed load of an atomic epoch and a compare. Only when
the epoch has moved -- twice per batch, a batch every few seconds -- does the
reader copy its pins into a buffer, behind a `try_lock` that Tracking never
waits for. A batch passes a grace period when every online reader has announced
since it began. A Global BA never announces, so no batch that begins a grace
period after it came online passes that grace period until it is over: its
snapshot of the whole map contains everything culled meanwhile, and
`FullInertialBA` does not ask `isBad()`. (A batch already past its retire when
the BA started may still be freed -- everything in it was unlinked before the
snapshot was taken.)

**Check.** After the first grace period the batch is checked against every
structure that can outlive a loop iteration, by looking, not by argument:

- every keyframe's slots, in every map, in Local Mapping's queue, and the
  culled keyframes that still have slots. Each keyframe's slot vector is copied
  under its mutex (16 KB, the same hold as `GetMapPointMatches()`), and probed
  against the batch outside the lock;
- the replaced-by link of every retired point not in the batch;
- the pins;
- `Map::mvpReferenceMapPoints` of every map.

A point found anywhere is **deferred**: it goes back, with a back-off, and is
held against the keyframe that names it so that it is not looked for again
until that keyframe is culled. Nothing in the SLAM state is written. A bad
point in a slot goes on blocking triangulation at that keypoint, Tracking goes
on using a just-culled point for one more frame, `CorrectLoop` goes on calling
`Replace()` on whatever is in the slot -- exactly as in v1.0, by construction.

**Free.** A second grace period lets the locals die that copied a slot before
the check. Then the batch is deleted, in slices.

The work runs on the Local Mapping thread inside the 3 ms it already sleeps at
the bottom of its loop, in slices of at most 1 ms. When Local Mapping is
stopped, Tracking is idle or a Global BA runs, reclamation waits; it is never
worse than today.

### Why it is safe

A point that has been bad for a full grace period and is pinned nowhere cannot
be written into a shared structure again: every path that writes a `MapPoint*`
takes it from an `isBad()`-guarded read in the same iteration, from a pinned
member, or from the replaced-by link of a point that is itself held. After the
first grace period the set of places naming a batch member only shrinks, the
check finds all of them, and the second grace period covers the copies taken
before the check.

What this rests on is the list of *members* that survive a thread's quiescent
point. A new long-lived pointer member that is not added to its thread's pins
is a use-after-free. The audit build (below) exists to catch that mechanically.

## Order of work

Each step is a commit that can be proven on its own.

1. Fixes that need no reclaimer: the `MLPnPsolver`s `Relocalization` never
   frees; vectors that are written and never read.

   Not among them, though every design and two reviewers put it first: the
   three places Local Mapping `delete`s the keyframes left in its queue, which
   Tracking made and still points at. Counted over a stereo-inertial, a
   monocular-inertial and a stereo run, all 25 purges found the queue empty.
   `InitializeIMU` takes the map-update mutex, which keeps Tracking out of
   `Track()`, and processes the queue before it purges; `Release()` is covered
   by `SetNotStop()`, which refuses a keyframe once Local Mapping has stopped.
   Only `ScaleRefinement` has a window -- the length of its inertial
   optimisation, between processing the queue and taking the mutex -- and it
   was not hit in 13 calls. Those sites become a retire when keyframes are
   reclaimed; until a run shows otherwise they are not a bug being lived with.
2. `Reclaimer`, owned by `Atlas`, wired into every `Map`, unused. Unit test
   with synthetic readers under ThreadSanitizer. objcmp: nothing else changes.
3. **Count**: retire wiring, nothing freed. At shutdown the graveyard must
   equal the memory report's "culled, never freed" rows.
4. **Dry run**: announces, pins, the batch state machine; nothing checked or
   freed. The A/B against v1.0 must be flat -- this is the only cost the design
   ever adds to a frame.
5. **Audit**: the check runs read-only with a batch threshold of zero; where it
   would free, it stops the world and scans every holder, and aborts naming the
   holder if a batch member is found. Run over KITTI with loops and Global BA,
   EuRoC inertial with resets, a merge, localisation mode, save and load.
6. **Poison**, under AddressSanitizer: instead of `delete`, fill the block with
   0xDD and poison it, so that no address is reused and any later touch reports
   with a stack. Then real `delete` under ASan.
7. On: `ORBSLAM3R_RECLAIM=points`, then the default once the A/B is flat and
   the culled-MapPoints row is near zero.
8. KeyFrames, designed against the census.
9. After that, the live objects: `sizeof(MapPoint)` (three mutexes, two maps
   used only while saving), the descriptor held as a `cv::Mat`, and the 16 KB of
   `mvuRight`/`mvDepth` a monocular keyframe carries for nothing.

## What was rejected, and why

- **A sentinel written into stale slots.** Three reviewers independently found
  the same hole: `CorrectLoop` calls `Replace()` on whatever is in the current
  keyframe's slot without asking `isBad()` (`LoopClosing.cpp`, "Update matched
  map points and replace if duplicated"). On a shared sentinel that writes into
  it and dereferences its null map. The census then showed the sentinel was
  solving a problem one slot in a thousand has.
- **Releasing a culled keyframe's descriptors and BoW vector in the same step
  that erases it from the database.** `KeyFrameDatabase` hands out keyframe
  pointers without asking `isBad()` and scores `pKFi->mBowVec` after dropping
  its mutex; unlink and free need a grace period between them, like the points.
- **Releasing a culled keyframe's keypoints and slots once no point in its own
  slots observes it.** The census shows live points observing a culled keyframe
  from outside its slots.
- **Draining the graveyard in `Shutdown()`.** `Shutdown()` also runs on the
  viewer thread (the Stop button) while the caller's thread is inside
  `Track()`; announcing for a thread that is not quiescent frees what it is
  using. It would also gain nothing: the process is ending.
- **Marking the objects of a wiped map bad.** `Tracking::UpdateFrameIMU` walks
  the spanning tree through `mlpReferences`; a wiped root that is suddenly bad
  has no parent, and the walk ends in a null pointer. Wiped units are retired
  in a state of their own.
- **A gate that stops the world.** It can block Tracking, and on a saturated
  device it starves. Announcing costs Tracking one load and never blocks it.
- **Reusing addresses without saying so.** MapPoint addresses will be reused,
  and `std::set<MapPoint*>` orders by address, so iteration order -- which
  already differs from run to run -- will differ in a new way. No bit-exact
  comparison of trajectories is possible, before or after; the acceptance test
  is the distribution of ATE and of per-frame time over repeated runs against
  v1.0.
