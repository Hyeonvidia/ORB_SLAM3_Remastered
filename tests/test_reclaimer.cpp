// Runs atlas/Reclaimer against threads that behave, in the ways that matter to
// it, like the ones in a SLAM system -- and against one that breaks the rule,
// to show that the test can tell.
//
//   test_reclaimer                 must pass
//   test_reclaimer --forget-pins   must fail: a reader keeps a pointer in a member
//                                  across its quiescent point and does not pin it
//
// The world: objects that are in a "map" (the live list) and in the slots of a
// few "keyframes". A writer creates them and culls them; culling takes an object
// out of the map, marks it bad, and takes it out of the slots that hold it --
// except that one time in ten it leaves a slot behind, as ORB-SLAM3 does.
// Readers copy slot vectors under a mutex and walk them, keep some of what they
// found in a member for the next iteration, and sometimes write a kept pointer
// into a slot, as Tracking does when a frame becomes a keyframe. One reader is
// a global bundle adjustment: it comes online, snapshots everything, works on
// the snapshot for a while without ever announcing, and goes offline.
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
        int payload = 0;
    };

    struct KeyFrame
    {
        std::mutex mutex;
        std::vector<Object*> slots = std::vector<Object*>(kSlots, nullptr);
    };

    std::mutex gMutexLive;
    std::vector<Object*> gLive;
    KeyFrame gKeyFrames[kKeyFrames];
    std::atomic<bool> gStop{false};
    std::atomic<long> gReads{0};

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

    void Reader(Reclaimer &reclaimer, Reclaimer::Reader id, unsigned seed, bool bPin)
    {
        std::mt19937 rng(seed);
        std::vector<Object*> held; // survives the loop, like Tracking's last frame
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

            // What it kept is still usable -- bad, perhaps, but there.
            for(Object* p : held)
                Touch(p, "a reader, through a member it kept");

            // Sometimes what it kept goes into a slot, bad or not.
            if(!held.empty() && rng() % 8 == 0)
            {
                KeyFrame &kf = gKeyFrames[rng() % kKeyFrames];
                std::lock_guard<std::mutex> lock(kf.mutex);
                kf.slots[rng() % kSlots] = held[rng() % held.size()];
            }

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
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        reclaimer.SetOnline(id, false);
    }

    void GlobalBA(Reclaimer &reclaimer)
    {
        while(!gStop.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            reclaimer.SetOnline(Reclaimer::GLOBAL_BA, true);
            std::vector<Object*> snapshot;
            {
                std::lock_guard<std::mutex> lock(gMutexLive);
                snapshot = gLive;
            }
            for(int k = 0; k < kKeyFrames; ++k)
            {
                const std::vector<Object*> copy = Slots(k);
                for(Object* p : copy)
                    if(p)
                        snapshot.push_back(p);
            }
            // Long after the snapshot, and without asking whether they are bad.
            for(int pass = 0; pass < 5; ++pass)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                for(Object* p : snapshot)
                    Touch(p, "the global BA, through its snapshot");
            }
            reclaimer.SetOnline(Reclaimer::GLOBAL_BA, false);
        }
    }

} // namespace

int main(int argc, char** argv)
{
    const bool bForgetPins = argc > 1 && std::strcmp(argv[1], "--forget-pins") == 0;

    std::vector<Object*> vpFreed;
    int nScanCursor = 0;
    Reclaimer::Hooks hooks;
    hooks.scan = [&nScanCursor](const Reclaimer::PointerSet &candidates, std::vector<void*> &vpNamed, bool bRestart,
                                std::chrono::microseconds budget)
    {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        if(bRestart)
            nScanCursor = 0;
        for(; nScanCursor < kKeyFrames; ++nScanCursor)
        {
            if(std::chrono::steady_clock::now() - start > budget)
                return false;
            const std::vector<Object*> copy = Slots(nScanCursor);
            for(Object* p : copy)
                if(p && candidates.Contains(p))
                    vpNamed.push_back(p);
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
    Reclaimer reclaimer(hooks, options);

    std::thread tracking(Reader, std::ref(reclaimer), Reclaimer::TRACKING, 1u, !bForgetPins);
    std::thread loopClosing(Reader, std::ref(reclaimer), Reclaimer::LOOP_CLOSING, 2u, true);
    std::thread viewer(Reader, std::ref(reclaimer), Reclaimer::VIEWER, 3u, true);
    std::thread gba(GlobalBA, std::ref(reclaimer));

    // This thread is the writer and the driver, as Local Mapping is.
    std::mt19937 rng(7);
    reclaimer.SetOnline(Reclaimer::LOCAL_MAPPING, true);
    std::vector<Object*> recent; // its own member, pinned
    long nCreated = 0, nCulled = 0, nStale = 0;
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now() +
                                                      std::chrono::seconds(bForgetPins ? 20 : 4);
    while(std::chrono::steady_clock::now() < end)
    {
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
                std::lock_guard<std::mutex> lock(kf.mutex);
                kf.slots[rng() % kSlots] = p;
            }
            recent.push_back(p);
        }
        for(int n = 0; n < 30; ++n)
        {
            Object* p = nullptr;
            {
                std::lock_guard<std::mutex> lock(gMutexLive);
                if(gLive.size() < 512)
                    break;
                const std::size_t i = rng() % gLive.size();
                p = gLive[i];
                gLive[i] = gLive.back();
                gLive.pop_back();
            }
            p->bad.store(true);
            const bool bLeaveOne = rng() % 10 == 0;
            bool bLeft = false;
            for(KeyFrame &kf : gKeyFrames)
            {
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
            ++nCulled;
            reclaimer.Retire(p);
        }
        if(recent.size() > 256)
            recent.erase(recent.begin(), recent.begin() + 128);
        for(Object* p : recent)
            Touch(p, "the writer, through its recent list");

        reclaimer.Announce(Reclaimer::LOCAL_MAPPING, [&recent](std::vector<const void*> &pins)
                           { pins.insert(pins.end(), recent.begin(), recent.end()); });
        reclaimer.Step(std::chrono::microseconds(1000));
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    gStop.store(true);
    tracking.join();
    loopClosing.join();
    viewer.join();
    gba.join();

    const Reclaimer::Stats stats = reclaimer.GetStats();
    std::printf("created %ld, culled %ld (%ld left in a slot), reads %ld\n", nCreated, nCulled, nStale, gReads.load());
    std::printf("retired %llu, freed %llu, waiting %zu (peak %zu), batches %llu\n",
                static_cast<unsigned long long>(stats.nRetired), static_cast<unsigned long long>(stats.nFreed),
                stats.nWaiting, stats.nPeakWaiting, static_cast<unsigned long long>(stats.nBatches));
    std::printf("put off: pinned %llu, named by a slot %llu, late pins %llu; longest step %lld us\n",
                static_cast<unsigned long long>(stats.nPutOffPinned),
                static_cast<unsigned long long>(stats.nPutOffNamed), static_cast<unsigned long long>(stats.nLatePins),
                static_cast<long long>(stats.longestStep.count()));

    int rc = 0;
    const auto expect = [&rc](bool ok, const char* what)
    {
        if(!ok)
        {
            std::fprintf(stderr, "FAILED: %s\n", what);
            rc = 1;
        }
    };
    expect(stats.nRetired == static_cast<std::uint64_t>(nCulled), "every culled object was retired");
    expect(stats.nFreed + stats.nWaiting == stats.nRetired, "retired = freed + still waiting");
    expect(stats.nFreed > stats.nRetired / 2, "most of what was culled was freed");
    expect(stats.nPutOffNamed > 0, "objects left in a slot were found by the check and put off");
    expect(stats.nPutOffPinned > 0, "objects a reader kept were put off");
    expect(stats.nLatePins == 0, "nothing was pinned after the check that was not pinned before it");
    expect(stats.longestStep < std::chrono::milliseconds(20), "a step stays near its budget");
    for(Object* p : vpFreed)
        delete p;
    std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
