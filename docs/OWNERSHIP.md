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

| culled KeyFrames named by (MH01) | as first measured | after `c4c6201` |
|---|---:|---:|
| a live point's observations | 40.7 % | 0 |
| a live keyframe's covisibility list | 0.2 % | 1.1 % |
| none of these | 59.1 % | 98.9 % |

As first measured, 323 of those observations were by a live point that was
*not* in the culled keyframe's own slot, and that looked like a permanent
fact about the system. It was a bug. `ORBmatcher::SearchForTriangulation`
declared `vbMatched2`, tested it, and never set it, so several features of one
keyframe could take the same feature of its neighbour; `CreateNewMapPoints`
made a point for each and wrote them into the one slot in turn, and every
point but the last kept an observation whose slot held another point. Since a
culled keyframe erases itself from its points' observations by walking its own
slots, those points never heard of the cull. Found with the slot audit
(`ORBSLAM3R_SLOT_AUDIT=1`, one line per mismatched pair with the mapping stage
it first appeared after: all after `CreateNewMapPoints`), fixed in one line,
and the census now reads zero for both rows.

So:

1. **A culled point that somebody still names can simply wait.** Stale slots in
   live keyframes are one in a thousand. There is no need to write anything
   into a slot in the point's place, which is what makes the rest simple.
2. **MapPoints first.** They are the part that grows with every frame, and on
   KITTI they are all of it.
3. **A culled keyframe's payload cannot be released on the strength of its own
   slots alone.** `MapPoint::UpdateNormalAndDepth` reads, without asking
   `isBad()`, the pose of every keyframe in a point's observations and the
   keypoints of its reference keyframe, so a live point observing a culled
   keyframe is a reader of its payload. With the duplicate-point bug fixed
   that is a race-only residue rather than the steady state it first looked
   like -- but the check still has to look for it, not assume it away.
   KeyFrames get their own design, after the MapPoints are done and with this
   table in hand.

## MapPoints: retire, wait, check, free

**Retire.** `MapPoint::SetBadFlag` and `Replace` end in
`Map::RetireMapPoint(this)` instead of `Map::EraseMapPoint(this)`: erase from
the set as today, and then, only if the point was in the set, one `push_back`
into the Reclaimer's incoming list under its mutex. Set membership is what
makes retiring idempotent -- `SearchAndFuse` can `Replace` the same point
twice, and a second record for a point whose batch has already gone would be a
double free. It also means a point that was never in a map, or whose map was
wiped by `Map::clear()`, is never handed over: it leaks as in v1.0, which is
the state of wiped units until they get an answer of their own.
`Map::EraseMapPoint` stays for the map-to-map moves of a merge. A culled
keyframe goes the same way, `Map::RetireKeyFrame`, onto a list the map keeps:
keyframes are not freed, and their slots must stay findable (below).

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

- every keyframe's slots: the live ones of every map, the culled ones of every
  map (`Map::GetCulledKeyFrames()` -- a culled keyframe keeps its slots, and
  Loop Closing and Tracking go on reading them through the members that name
  it, `mpCurrentKF` from the loop queue, `mpReferenceKF`), and Local Mapping's
  queue. Each keyframe's slot vector is copied under its mutex (16 KB, the same
  hold as `GetMapPointMatches()`), and probed against the batch outside the
  lock;
- the replaced-by link of every retired point not in the batch, which the
  Reclaimer hands the scan as `waiting`: `Tracking::CheckReplacedInLastFrame`
  follows a pinned culled point to its replacement and puts the replacement in
  the last frame, and the replacement may have been culled since;
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
A pin that appears at the second grace period and was not there at the first
("late pin") is what an incomplete list looks like from the inside: it is put
off, never freed, and counted, and the count must be zero.

`System::GetTrackedMapPoints()` hands the last frame's pointers to the caller.
They are valid until the next `Track*()` call begins; a wrapper that keeps
them longer was safe under v1.0, which freed nothing, and is not now.

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
   *Done.* Three reviewers then read it against the real threads; what they
   found and what the test found afterwards is in the commits, and the test
   has a variant for each way the protocol can be broken.
3. **Count**: retire wiring, nothing freed. At shutdown the graveyard must
   equal the memory report's "culled, never freed" rows. *Done*: equal on
   KITTI 00 stereo and mono and on MH01 stereo- and mono-inertial (26 maps).
   The culled keyframes' list is checked the same way: 468 of 468.
4. **Dry run**: announces, pins, the batch state machine; nothing checked or
   freed. The A/B against v1.0 must be flat -- this is the only cost the design
   ever adds to a frame. *Done*: `tools/ab_stages.sh`, count against dry run,
   six pairs on three sequences, tracking +0.18, +0.01, -0.03 ms with either
   sign from pair to pair.
5. **Poison**, under AddressSanitizer: `ORBSLAM3R_RECLAIM=poison` runs the check
   and, instead of `delete`, destroys the point, fills its block with 0xDD and
   poisons it, so that no address is reused and any later touch is a report
   with the stack that made it. With `ORBSLAM3R_RECLAIM_BATCH=64` a batch is
   every 64 culls rather than every 8192, so that whatever can go wrong goes
   wrong often. Run over KITTI stereo with loops and a Global BA, EuRoC
   stereo-inertial, and mono-inertial, which under the sanitizer's slowdown
   loses tracking, resets and merges over and over.

   This replaces the stop-the-world audit planned here: the report from a
   poisoned access names the holder as well as an audit would, and needs no
   mechanism of its own. Late pins are the other audit and must stay at zero.
   *Done*: no report and no late pin in any of the five --
   KITTI 04 and 07 stereo, KITTI 00 stereo (a loop closure and its Global BA;
   446,663 points freed in 74 batches), MH01 stereo-inertial (371 batches) and
   MH01 mono-inertial, which under the sanitizer reset 211 times. The one
   abort was v1.0's, after Shutdown(): the trajectory savers left their map
   pointer uninitialised when every map was empty. Fixed.
6. On: `ORBSLAM3R_RECLAIM=points`, the default once the A/B is flat and the
   culled-MapPoints row is near zero. *The default.* It was not at first: the
   full KITTI A/B against v1.0 (110 runs) put stereo tracking at +3.3 %, slower
   in 10 sequences of 11, where it had been -0.5 % before freeing; accuracy
   unchanged. Where it went, measured with one binary under the same
   four-at-a-time load: not the announces (count against dry is flat), not
   the scan (four times bigger batches, a quarter of the scans, cost the
   same), but the two stages that walk the current frame's and the local
   map's points, Pose Prediction and LM Track, +0.1 to +0.15 ms each on every
   pair -- and nothing in `poison`, which frees the members but never the
   block. glibc's knobs do not help (trim thresholds cost 700-900 MB of RSS).
   Two changes took it away. `atlas/SlotPool`: MapPoints come from slabs of
   their own, consecutive allocations filling one slab, so new points stay
   together whatever has been freed and freeing never reaches glibc. That
   halved nothing measurable by itself. The descriptor: each point held its 32
   bytes in a cv::Mat with its own heap block, and freeing 500,000 of those
   scattered glibc's small-chunk bins across the heap, where Tracking's
   per-frame vectors and local BA's g2o objects were then placed. With the 32
   bytes inside the point, dry against points is -0.29, -0.08 and -0.38 ms of
   tracking on KITTI 03, 04 and 07 stereo -- freeing is now slightly the
   faster of the two -- and local BA is flat. The full KITTI A/B against v1.0,
   run again with it all on: stereo -0.6 % per frame, faster in 9 sequences
   of 11; mono 0.0 %, either sign; accuracy the same within the spread. What
   freeing does to the memory report, run against the dry run on the same
   machine at the same time:

   | | culled MapPoints | peak RSS |
   |---|---:|---:|
   | KITTI 00 stereo | 415 MB -> 5.7 MB | 2160 -> 1676 MB |
   | KITTI 00 mono | 258 MB -> 8.1 MB | 3557 -> 3087 MB |
   | MH01 stereo-inertial | 93 MB -> 18 MB | 515 -> 424 MB |
   | MH01 mono-inertial | 85 MB -> 11 MB | 656 -> 537 MB |

   What stays is what the census said would stay: on MH01 the culled points
   that only culled keyframes' slots still name (19 %), which step 7 releases,
   and on all of them the few thousand that were in flight at the end.
7. KeyFrames, designed against the census.
8. After that, the live objects: `sizeof(MapPoint)` (three mutexes, two maps
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
