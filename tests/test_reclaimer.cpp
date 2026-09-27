// Runs atlas/Reclaimer against threads that behave, in the ways that matter to
// it, like the ones in a SLAM system -- and against ones that break a rule, to
// show that the test can tell.
//
//   test_reclaimer                    must pass
//   test_reclaimer --dry-run          must pass: the protocol runs, nothing is freed
//   test_reclaimer --forget-pins      must fail: a reader keeps a pointer in a member
//                                     across its quiescent point and does not pin it
//   test_reclaimer --forget-culled    must fail: the check does not look in the slots
//                                     of culled keyframes, which readers go on reading
//   test_reclaimer --forget-replaced  must fail: the check does not follow the
//                                     replaced-by links of retired objects, which a
//                                     reader follows from a pinned one
//
// The world: objects that are in a "map" (the live list) and in the slots of a
// few "keyframes". The writer creates them and culls them; culling takes an
// object out of the map, marks it bad, and takes it out of the slots of live
// keyframes -- not of culled ones, which keep their slots, as ORB-SLAM3's do --
// and one time in ten leaves a slot behind even in a live one. Half the culled
// objects name a live object as their replacement. The writer also culls
// keyframes, which readers go on reading for a while through an index they
// keep. Readers copy slot vectors under a mutex and walk them, keep some of
// what they found in a member for many iterations, follow the replacement of a
// culled object they hold and keep that too, and sometimes write a kept pointer
// into a slot, as Tracking does when a frame becomes a keyframe. One reader
// culls objects itself, as Loop Closing does, and does so while the driver is
// parked. A global bundle adjustment is put online by the writer's thread and
// taken offline by its own: it snapshots everything, works on the snapshot for
// a while without ever announcing, and sometimes leaves early.
//
// Freed objects are not deleted until the end: they are marked dead, and every
// access checks the mark. A use-after-free is then a certain abort rather than
// a read of memory that may happen to be intact.
#include "atlas/Reclaimer.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

namespace
{

    using ORB_SLAM3::Reclaimer;

    const int kAlive = 0x600D, kDead = 0xDEAD;
    const int kKeyFrames = 24, kSlots = 256;

    struct Object
    {
        std::atomic<int> state{kAlive};
        std::atomic<bool> bad{false};
        std::atomic<Object*> replaced{nullptr};
        int payload = 0;
    };

    struct KeyFrame
    {
        std::mutex mutex;
        std::vector<Object*> slots = std::vector<Object*>(kSlots, nullptr);
        std::atomic<bool> culled{false};
        std::chrono::steady_clock::time_point culledAt;
    };

    std::mutex gMutexLive;
    std::vector<Object*> gLive;
    KeyFrame gKeyFrames[kKeyFrames];
    std::atomic<bool> gStop{false};
    std::atomic<long> gReads{0};
    std::atomic<int> gGbaRequested{0};

    void Touch(const Object* p, const char* who)
    {
        if(p->state.load() != kAlive)
        {
            std::fprintf(stderr, "USE AFTER FREE: %s read a freed object\n", who);
            std::_Exit(3);
        }
        gReads.fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<Object*> Slots(int k)
    {
        std::lock_guard<std::mutex> lock(gKeyFrames[k].mutex);
        return gKeyFrames[k].slots;
    }

    // What MapPoint::SetBadFlag() and Replace() do, then Map::RetireMapPoint().
    void Cull(Reclaimer &reclaimer, std::mt19937 &rng, long &nStale)
    {
        Object* p = nullptr;
        Object* pReplacement = nullptr;
        {
            std::lock_guard<std::mutex> lock(gMutexLive);
            if(gLive.size() < 512)
                return;
            const std::size_t i = rng() % gLive.size();
            p = gLive[i];
            gLive[i] = gLive.back();
            gLive.pop_back();
            if(rng() % 4 != 0)
                pReplacement = gLive[rng() % gLive.size()];
        }
        p->bad.store(true);
        p->replaced.store(pReplacement);
        const bool bLeaveOne = rng() % 10 == 0;
        bool bLeft = false;
        for(KeyFrame &kf : gKeyFrames)
        {
            if(kf.culled.load())
                continue; // a culled keyframe is out of the observations: its slots stay
            std::lock_guard<std::mutex> lock(kf.mutex);
            for(Object*&slot : kf.slots)
                if(slot == p)
                {
                    if(bLeaveOne && !bLeft)
                        bLeft = true;
                    else
                        slot = nullptr;
                }
        }
        nStale += bLeft;
        reclaimer.Retire(p);
    }

    void Reader(Reclaimer &reclaimer, Reclaimer::Reader id, unsigned seed, bool bPin)
    {
        std::mt19937 rng(seed);
        std::vector<Object*> held; // survives the loop, like Tracking's last frame
        int heldKF = -1;           // like Loop Closing's current keyframe, taken from its queue
        long nCulled = 0, nStale = 0;
        reclaimer.SetOnline(id, true);
        while(!gStop.load())
        {
            reclaimer.Announce(
                id,
                [&held, bPin](std::vector<const void*> &pins)
                {
                    if(bPin)
                        pins.insert(pins.end(), held.begin(), held.end());
                },
                id != Reclaimer::TRACKING);

            // What it kept is still usable -- bad, perhaps, but there. Now and
            // then a culled one that names a replacement is followed there, and
            // the replacement kept: Tracking::CheckReplacedInLastFrame, once a
            // frame, which is many batches of culling later than the cull. A
            // variant that must fail runs until it does, up to 40 s: this one's
            // chance per batch is small.
            const bool bFollow = id == Reclaimer::TRACKING && rng() % 16 == 0;
            const std::size_t nHeld = held.size();
            for(std::size_t i = 0; i < nHeld; ++i)
            {
                Touch(held[i], "a reader, through a member it kept");
                Object* pRep = bFollow && held[i]->bad.load() ? held[i]->replaced.load() : nullptr;
                if(pRep)
                {
                    Touch(pRep, "a reader, through a replaced-by link");
                    held.push_back(pRep);
                }
            }

            // Sometimes what it kept goes into a slot, bad or not.
            if(!held.empty() && rng() % 8 == 0)
            {
                KeyFrame &kf = gKeyFrames[rng() % kKeyFrames];
                std::lock_guard<std::mutex> lock(kf.mutex);
                kf.slots[rng() % kSlots] = held[rng() % held.size()];
            }

            // The keyframe it keeps is read whether or not it has been culled since.
            if(heldKF < 0 || rng() % 64 == 0)
                heldKF = rng() % kKeyFrames;
            for(Object* p : Slots(heldKF))
                if(p)
                    Touch(p, "a reader, through the slots of a keyframe it keeps");

            // What it keeps, it keeps for many iterations -- a reference keyframe,
            // not a last frame. Two grace periods would cover a pointer kept for
            // one iteration even if it were never pinned.
            const bool bRenew = rng() % 32 == 0;
            if(bRenew)
                held.clear();
            for(int n = 0; n < 4; ++n)
            {
                const std::vector<Object*> copy = Slots(rng() % kKeyFrames);
                for(Object* p : copy)
                {
                    if(!p)
                        continue;
                    Touch(p, "a reader, through a slot");
                    if(p->bad.load())
                        continue;
                    if(bRenew && rng() % 64 == 0)
                        held.push_back(p);
                }
            }

            // Loop Closing culls too (SearchAndFuse), and it does so while the
            // driver is parked.
            if(id == Reclaimer::LOOP_CLOSING && rng() % 4 == 0)
            {
                Cull(reclaimer, rng, nStale);
                ++nCulled;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        reclaimer.SetOnline(id, false);
    }

    void GlobalBA(Reclaimer &reclaimer)
    {
        int nDone = 0;
        std::mt19937 rng(99);
        while(!gStop.load())
        {
            if(gGbaRequested.load() <= nDone)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            // Online since before this thread was told to go; offline at whichever
            // exit it takes.
            struct Offline
            {
                Reclaimer &r;
                ~Offline() { r.SetOnline(Reclaimer::GLOBAL_BA, false); }
            } offline{reclaimer};

            std::vector<Object*> snapshot;
            {
                std::lock_guard<std::mutex> lock(gMutexLive);
                snapshot = gLive;
            }
            for(int k = 0; k < kKeyFrames; ++k)
                for(Object* p : Slots(k))
                    if(p)
                        snapshot.push_back(p);
            // Long after the snapshot, and without asking whether they are bad.
            const int nPasses = rng() % 3 == 0 ? 1 : 5;
            for(int pass = 0; pass < nPasses; ++pass)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                for(Object* p : snapshot)
                    Touch(p, "the global BA, through its snapshot");
            }
            ++nDone;
        }
    }

} // namespace

int main(int argc, char** argv)
{
    const char* arg = argc > 1 ? argv[1] : "";
    const bool bForgetPins = std::strcmp(arg, "--forget-pins") == 0;
    const bool bForgetCulled = std::strcmp(arg, "--forget-culled") == 0;
    const bool bForgetReplaced = std::strcmp(arg, "--forget-replaced") == 0;
    const bool bDryRun = std::strcmp(arg, "--dry-run") == 0;
    const bool bMustFail = bForgetPins || bForgetCulled || bForgetReplaced;

    std::vector<Object*> vpFreed;
    int nScanCursor = 0;
    Reclaimer::Hooks hooks;
    hooks.scan = [&nScanCursor, bForgetCulled,
                  bForgetReplaced](const Reclaimer::PointerSet &candidates, const std::vector<void*> &waiting,
                                   std::vector<void*> &vpNamed, bool bRestart, std::chrono::microseconds budget)
    {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        if(bRestart)
            nScanCursor = 0;
        for(; nScanCursor < kKeyFrames; ++nScanCursor)
        {
            if(std::chrono::steady_clock::now() - start > budget)
                return false;
            if(bForgetCulled && gKeyFrames[nScanCursor].culled.load())
                continue;
            for(Object* p : Slots(nScanCursor))
                if(candidates.Contains(p))
                    vpNamed.push_back(p);
        }
        if(!bForgetReplaced)
        {
            for(void* w : waiting)
            {
                Object* pRep = static_cast<Object*>(w)->replaced.load();
                if(candidates.Contains(pRep))
                    vpNamed.push_back(pRep);
            }
            // A named candidate is kept, so what it names is kept too: a chain
            // Z -> X -> Y with Z pinned and X, Y both in this batch.
            std::unordered_set<void*> named(vpNamed.begin(), vpNamed.end());
            for(std::size_t i = 0; i < vpNamed.size(); ++i)
            {
                Object* pRep = static_cast<Object*>(vpNamed[i])->replaced.load();
                if(candidates.Contains(pRep) && named.insert(pRep).second)
                    vpNamed.push_back(pRep);
            }
        }
        return true;
    };
    hooks.destroy = [&vpFreed](void* p)
    {
        Object* pObject = static_cast<Object*>(p);
        pObject->state.store(kDead);
        vpFreed.push_back(pObject);
    };
    Reclaimer::Options options;
    options.nBatchObjects = 256;
    options.maxAge = std::chrono::seconds(1);
    options.bCheckAndFree = !bDryRun;
    Reclaimer reclaimer(hooks, options);

    std::thread tracking(Reader, std::ref(reclaimer), Reclaimer::TRACKING, 1u, !bForgetPins);
    std::thread loopClosing(Reader, std::ref(reclaimer), Reclaimer::LOOP_CLOSING, 2u, true);
    std::thread viewer(Reader, std::ref(reclaimer), Reclaimer::VIEWER, 3u, true);
    std::thread gba(GlobalBA, std::ref(reclaimer));

    // This thread is the writer and the driver, as Local Mapping is.
    std::mt19937 rng(7);
    reclaimer.SetOnline(Reclaimer::LOCAL_MAPPING, true);
    std::vector<Object*> recent; // its own member, pinned
    long nCreated = 0, nCulled = 0, nStale = 0, nKeyFramesCulled = 0, nIterations = 0;
    std::chrono::steady_clock::time_point lastGBA = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now() +
                                                      std::chrono::seconds(bMustFail ? 40 : 4);
    while(std::chrono::steady_clock::now() < end)
    {
        ++nIterations;
        for(int n = 0; n < 32; ++n)
        {
            Object* p = new Object;
            p->payload = static_cast<int>(nCreated++);
            {
                std::lock_guard<std::mutex> lock(gMutexLive);
                gLive.push_back(p);
            }
            for(int s = 0; s < 3; ++s)
            {
                KeyFrame &kf = gKeyFrames[rng() % kKeyFrames];
                if(kf.culled.load())
                    continue;
                std::lock_guard<std::mutex> lock(kf.mutex);
                kf.slots[rng() % kSlots] = p;
            }
            recent.push_back(p);
        }
        for(int n = 0; n < 30; ++n)
        {
            Cull(reclaimer, rng, nStale);
            ++nCulled;
        }
        // Keyframes are culled, and -- this world being finite -- brought back
        // empty much later, long after whoever kept an index has read it.
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if(nIterations % 50 == 0)
        {
            KeyFrame &kf = gKeyFrames[rng() % kKeyFrames];
            if(!kf.culled.load())
            {
                kf.culledAt = now;
                kf.culled.store(true);
                ++nKeyFramesCulled;
            }
        }
        for(KeyFrame &kf : gKeyFrames)
            if(kf.culled.load() && now - kf.culledAt > std::chrono::milliseconds(300))
            {
                std::lock_guard<std::mutex> lock(kf.mutex);
                for(Object*&slot : kf.slots)
                    slot = nullptr;
                kf.culled.store(false);
            }
        if(now - lastGBA > std::chrono::milliseconds(150))
        {
            lastGBA = now;
            reclaimer.SetOnline(Reclaimer::GLOBAL_BA, true); // from this thread, as Loop Closing does
            gGbaRequested.fetch_add(1);
        }
        if(recent.size() > 256)
            recent.erase(recent.begin(), recent.begin() + 128);
        for(Object* p : recent)
            Touch(p, "the writer, through its recent list");

        reclaimer.Announce(Reclaimer::LOCAL_MAPPING, [&recent](std::vector<const void*> &pins)
                           { pins.insert(pins.end(), recent.begin(), recent.end()); });
        reclaimer.Step(std::chrono::microseconds(1000));
        // Parked now and then, as Local Mapping is for a loop correction, while
        // Loop Closing goes on culling.
        std::this_thread::sleep_for(nIterations % 200 == 0 ? std::chrono::microseconds(30000)
                                                           : std::chrono::microseconds(500));
    }
    // The run is over, nothing is retired any more, and the readers go on.
    // What was put off must still come back and be freed.
    const std::size_t nWaitingAtEnd = reclaimer.GetStats().nWaiting;
    recent.clear();
    for(int n = 0; n < 2500; ++n)
    {
        reclaimer.Announce(Reclaimer::LOCAL_MAPPING, [](std::vector<const void*> &) {});
        reclaimer.Step(std::chrono::microseconds(1000));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    gStop.store(true);
    tracking.join();
    loopClosing.join();
    viewer.join();
    gba.join();

    const Reclaimer::Stats stats = reclaimer.GetStats();
    std::printf("created %ld, culled %ld by the writer (%ld left in a slot) and %llu in all, %ld keyframes culled, "
                "reads %ld\n",
                nCreated, nCulled, nStale, static_cast<unsigned long long>(stats.nRetired), nKeyFramesCulled,
                gReads.load());
    std::printf("retired %llu, freed %llu, kept %llu, waiting %zu (peak %zu), batches %llu\n",
                static_cast<unsigned long long>(stats.nRetired), static_cast<unsigned long long>(stats.nFreed),
                static_cast<unsigned long long>(stats.nKept), stats.nWaiting, stats.nPeakWaiting,
                static_cast<unsigned long long>(stats.nBatches));
    std::printf("put off: pinned %llu, named %llu, late pins %llu; longest step %lld us\n",
                static_cast<unsigned long long>(stats.nPutOffPinned),
                static_cast<unsigned long long>(stats.nPutOffNamed), static_cast<unsigned long long>(stats.nLatePins),
                static_cast<long long>(stats.longestStep.count()));
    std::printf("waiting when the run ended %zu, after idling %zu\n", nWaitingAtEnd, stats.nWaiting);

    int rc = 0;
    const auto expect = [&rc](bool ok, const char* what)
    {
        if(!ok)
        {
            std::fprintf(stderr, "FAILED: %s\n", what);
            rc = 1;
        }
    };
    expect(stats.nRetired >= static_cast<std::uint64_t>(nCulled), "every culled object was retired");
    if(bDryRun)
    {
        expect(stats.nFreed == 0 && vpFreed.empty(), "a dry run frees nothing");
        expect(stats.nKept > stats.nRetired / 2, "a dry run runs the protocol to its end");
    }
    else
    {
        expect(stats.nFreed > stats.nRetired / 2, "most of what was culled was freed");
        expect(stats.nPutOffNamed > 0, "objects a slot or a link named were found by the check and put off");
        expect(stats.nLatePins == 0, "nothing was pinned after the check that was not pinned before it");
    }
    expect(stats.nPutOffPinned > 0, "objects a reader kept were put off");
    expect(stats.nWaiting < nWaitingAtEnd / 4 + 64, "what was put off came back once nothing held it");
#if !defined(__SANITIZE_THREAD__)
    // Not under ThreadSanitizer, where every mutex costs what a scan costs here.
    expect(stats.longestStep < std::chrono::milliseconds(20), "a step stays near its budget");
#endif
    for(Object* p : vpFreed)
        delete p;
    std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
