/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef RECLAIMER_H
#define RECLAIMER_H

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ORB_SLAM3
{

    // Frees the objects a map has culled, once nobody can still be using them.
    // docs/OWNERSHIP.md has the design and why it is this one; in short:
    //
    //   retire   whoever culls an object hands it over. Nothing is freed then.
    //   wait     every reader -- a thread -- passes a point in its loop where it
    //            holds no pointer it got earlier, except in members, and
    //            publishes those members as pins.
    //   check    the owner looks in every structure that outlives a loop
    //            iteration. An object somebody still names is put off; nothing
    //            is written in its place.
    //   wait     again, for the copies taken before the check.
    //   free
    //
    // This class is the protocol and nothing else: it knows no MapPoint and no
    // KeyFrame. What to look in, and what freeing means, come from its owner as
    // hooks, which is also what lets it be tested with threads that are not a
    // SLAM system.
    //
    // A reader's cost when nothing is going on is Announce()'s first line: one
    // load of an atomic that only the driver writes, and a compare.
    class Reclaimer
    {
    public:
        enum Reader
        {
            TRACKING = 0,
            LOCAL_MAPPING,
            LOOP_CLOSING,
            VIEWER,
            GLOBAL_BA,
            N_READERS
        };

        // A set of addresses, built once per batch and then only asked.
        class PointerSet
        {
        public:
            void Build(const std::vector<void*> &vp);
            bool Contains(const void* p) const
            {
                // nullptr marks an empty slot, so it must be answered before the
                // probe: keyframe slots are mostly null.
                if(!p || mvSlots.empty())
                    return false;
                for(std::size_t i = Hash(p) & mnMask;; i = (i + 1) & mnMask)
                {
                    if(mvSlots[i] == p)
                        return true;
                    if(!mvSlots[i])
                        return false;
                }
            }
            std::size_t Size() const { return mnSize; }

        private:
            static std::size_t Hash(const void* p)
            {
                // Heap addresses are 16-aligned; fold the high bits down.
                const std::uintptr_t x = reinterpret_cast<std::uintptr_t>(p) >> 4;
                return static_cast<std::size_t>((x ^ (x >> 17)) * 0x9E3779B97F4A7C15ull >> 20);
            }
            std::vector<const void*> mvSlots;
            std::size_t mnMask = 0, mnSize = 0;
        };

        // Neither hook may throw. Both run on the driver's thread.
        struct Hooks
        {
            // Look in the long-lived holders for members of `candidates` and append
            // the ones found to `vpNamed`; nullptr entries are ignored. Called again
            // and again until it returns true (done); each call should take about
            // `budget`. `bRestart` is true on the first call for a batch.
            std::function<bool(const PointerSet &candidates, std::vector<void*> &vpNamed, bool bRestart,
                               std::chrono::microseconds budget)>
                scan;
            // Free one object.
            std::function<void(void*)> destroy;
        };

        struct Options
        {
            std::size_t nBatchObjects = 8192;     // start a batch at this many retired objects,
            std::chrono::seconds maxAge{10};      // or when the oldest has waited this long;
            std::size_t nMaxBatchObjects = 32768; // never more than this in one batch, so that the
                                                  // steps that walk a whole batch stay short after a
                                                  // long hold-up (a global BA, a wiped map)
            bool bCheckAndFree = true;            // false: run the protocol, keep the objects (a dry run)
        };

        enum class State
        {
            IDLE,
            GRACE1,
            CHECK,
            GRACE2,
            FREE
        };

        struct Stats
        {
            std::uint64_t nRetired = 0, nFreed = 0, nKept = 0, nBatches = 0;
            std::uint64_t nPutOffPinned = 0, nPutOffNamed = 0, nLatePins = 0;
            // As of the end of the last Step(): retired and neither freed nor kept.
            // nRetired == nFreed + nKept + nWaiting once everything is quiet.
            std::size_t nWaiting = 0, nPeakWaiting = 0;
            std::chrono::microseconds longestStep{0};
            // Where the driver is and for how long, so that a stall can be seen.
            State state = State::IDLE;
            std::chrono::steady_clock::time_point stateSince;
        };

        Reclaimer(Hooks hooks, Options options);
        ~Reclaimer();

        Reclaimer(const Reclaimer &) = delete;
        Reclaimer &operator=(const Reclaimer &) = delete;

        // Any thread. The caller guarantees an object is retired once.
        void Retire(void* p);

        // A reader exists from SetOnline(r, true) to SetOnline(r, false); while it
        // is online, no grace period that begins after it came online ends without
        // it. (One already under way may end: what that batch holds was retired,
        // and so unlinked, before the reader came online.) It must hold no pointer
        // to a retirable object when it comes online, and the call is made on the
        // reader's own thread at such a point -- except for a reader that never
        // announces, a global bundle adjustment working on a snapshot of
        // everything, which any thread may put online and which holds reclamation
        // up for as long as it is online. That is the intent.
        void SetOnline(Reader r, bool bOnline);

        // The reader's quiescent point. `pins` is called only when a grace period is
        // waiting for this reader, and appends every pointer the reader still holds
        // in members. With bWait false the announce is skipped, not waited for, if
        // the driver holds the lock, and tried again at the next call.
        template<class PinFn>
        void Announce(Reader r, PinFn &&pins, bool bWait = true)
        {
            Slot &slot = mSlots[r];
            const std::uint32_t epoch = mEpoch.load(std::memory_order_relaxed);
            if(epoch == slot.nSeen)
                return;
            slot.vpScratch.clear();
            pins(slot.vpScratch);
            Publish(r, epoch, bWait);
        }

        // The driver: one thread, the same one every time (asserted in debug
        // builds). Does about `budget` of work and returns.
        void Step(std::chrono::microseconds budget);

        Stats GetStats() const;

    private:
        struct alignas(64) Slot
        {
            // The reader's own thread's; nobody else touches them.
            std::uint32_t nSeen = 0;
            std::vector<const void*> vpScratch;
            // Under mMutexReaders.
            bool bOnline = false;
            std::uint32_t nAnnounced = 0;
            std::vector<const void*> vpPins;
        };

        struct Entry
        {
            void* p;
            std::uint8_t nPutOff;
            // When retired; for one that was put off, when it is to be looked at again.
            std::chrono::steady_clock::time_point when;
        };

        void Publish(Reader r, std::uint32_t epoch, bool bWait);
        bool GraceOver(std::uint32_t epoch, std::vector<const void*> &vpPins);
        void PutOffAll(const std::vector<const void*> &vp, std::uint64_t &counter,
                       std::chrono::steady_clock::time_point now);
        void Enter(State state, std::chrono::steady_clock::time_point now);

        const Hooks mHooks;
        const Options mOptions;

        alignas(64) std::atomic<std::uint32_t> mEpoch{1};
        std::array<Slot, N_READERS> mSlots;

        std::mutex mMutexReaders;

        mutable std::mutex mMutexIncoming;
        std::vector<Entry> mvIncoming;
        std::uint64_t mnRetired = 0;

        // The driver's.
        State mState = State::IDLE;
        std::chrono::steady_clock::time_point mStateSince;
        std::uint32_t mnGraceEpoch = 0;
        std::vector<Entry> mvBatch, mvPutOff;
        std::vector<void*> mvpCandidates, mvpNamed;
        PointerSet mCandidates;
        bool mbScanRestart = true;
        std::size_t mnFreeCursor = 0;
        std::thread::id mDriver;

        mutable std::mutex mMutexStats;
        Stats mStats;
    };

} // namespace ORB_SLAM3

#endif // RECLAIMER_H
