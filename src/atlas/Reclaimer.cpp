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

#include "atlas/Reclaimer.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace ORB_SLAM3
{

    namespace
    {
        // An object that was put off n times is looked at again after maxAge * 2^n,
        // up to this: one that a live keyframe names may be named for the rest of
        // the run, and must not bring the whole map's scan with it every time.
        const unsigned kMaxPutOffShift = 6;
    } // namespace

    void Reclaimer::PointerSet::Build(const std::vector<void*> &vp)
    {
        std::size_t nSlots = 16;
        while(nSlots < vp.size() * 2 + 1)
            nSlots *= 2;
        mvSlots.assign(nSlots, nullptr);
        mnMask = nSlots - 1;
        mnSize = 0;
        for(const void* p : vp)
        {
            if(!p)
                continue;
            std::size_t i = Hash(p) & mnMask;
            while(mvSlots[i] && mvSlots[i] != p)
                i = (i + 1) & mnMask;
            if(!mvSlots[i])
            {
                mvSlots[i] = p;
                ++mnSize;
            }
        }
    }

    Reclaimer::Reclaimer(Hooks hooks, Options options) : mHooks(std::move(hooks)), mOptions(options)
    {
        assert(!mOptions.bCheckAndFree || (mHooks.scan && mHooks.destroy));
        mStateSince = std::chrono::steady_clock::now();
        mStats.stateSince = mStateSince;
    }

    Reclaimer::~Reclaimer()
    {
        // Whatever is still here is freed by nobody, as before there was a
        // Reclaimer: the threads that might hold it are not this class's to join.
    }

    void Reclaimer::Retire(void* p)
    {
        if(!p)
            return;
        const Entry entry{p, 0, std::chrono::steady_clock::now()};
        std::lock_guard<std::mutex> lock(mMutexIncoming);
        mvIncoming.push_back(entry);
        ++mnRetired;
    }

    void Reclaimer::SetOnline(Reader r, bool bOnline)
    {
        std::lock_guard<std::mutex> lock(mMutexReaders);
        Slot &slot = mSlots[r];
        slot.bOnline = bOnline;
        slot.vpPins.clear();
        // It holds nothing now, which is what announcing says. nSeen is the
        // reader's thread's own and is left alone: if it is behind, the reader
        // publishes once more at its next Announce, which changes nothing.
        slot.nAnnounced = mEpoch.load();
    }

    void Reclaimer::Publish(Reader r, std::uint32_t epoch, bool bWait)
    {
        Slot &slot = mSlots[r];
        std::unique_lock<std::mutex> lock(mMutexReaders, std::defer_lock);
        if(bWait)
            lock.lock();
        else if(!lock.try_lock())
            return; // nSeen is unchanged, so the next call comes back here
        slot.vpPins.swap(slot.vpScratch);
        slot.nAnnounced = epoch;
        slot.nSeen = epoch;
    }

    bool Reclaimer::GraceOver(std::uint32_t epoch, std::vector<const void*> &vpPins)
    {
        std::lock_guard<std::mutex> lock(mMutexReaders);
        for(const Slot &slot : mSlots)
            if(slot.bOnline && slot.nAnnounced < epoch)
                return false;
        vpPins.clear();
        for(const Slot &slot : mSlots)
            if(slot.bOnline)
                vpPins.insert(vpPins.end(), slot.vpPins.begin(), slot.vpPins.end());
        return true;
    }

    // Takes out of the batch whatever `vp` names, counting it under `counter`.
    void Reclaimer::PutOffAll(const std::vector<const void*> &vp, std::uint64_t &counter,
                              std::chrono::steady_clock::time_point now)
    {
        if(vp.empty() || mvBatch.empty())
            return;
        std::vector<void*> vpNamed;
        vpNamed.reserve(vp.size());
        for(const void* p : vp)
            vpNamed.push_back(const_cast<void*>(p));
        PointerSet named;
        named.Build(vpNamed);
        for(Entry &entry : mvBatch)
            if(named.Contains(entry.p))
            {
                if(entry.nPutOff < 255)
                    ++entry.nPutOff;
                const unsigned shift = std::min<unsigned>(entry.nPutOff - 1, kMaxPutOffShift);
                entry.when = now + mOptions.maxAge * (1u << shift);
                mvPutOff.push_back(entry);
                entry.p = nullptr;
                ++counter;
            }
        mvBatch.erase(std::remove_if(mvBatch.begin(), mvBatch.end(), [](const Entry &e) { return !e.p; }),
                      mvBatch.end());
    }

    void Reclaimer::Enter(State state, std::chrono::steady_clock::time_point now)
    {
        mState = state;
        mStateSince = now;
    }

    void Reclaimer::Step(std::chrono::microseconds budget)
    {
        if(mDriver == std::thread::id())
            mDriver = std::this_thread::get_id();
        assert(mDriver == std::this_thread::get_id() && "Reclaimer::Step has one driver");

        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const auto spent = [&start]
        { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start); };
        Stats delta;

        if(mState == State::IDLE)
        {
            bool bBegin = false;
            {
                std::lock_guard<std::mutex> lock(mMutexIncoming);
                const bool bFull = !mvIncoming.empty() && mvIncoming.size() >= mOptions.nBatchObjects;
                const bool bOld = !mvIncoming.empty() && start - mvIncoming.front().when >= mOptions.maxAge;
                if(bFull || bOld)
                {
                    bBegin = true;
                    if(mvIncoming.size() <= mOptions.nMaxBatchObjects)
                        mvBatch.swap(mvIncoming);
                    else
                    {
                        // The oldest go first; the rest stay, and bOld keeps firing.
                        mvBatch.assign(mvIncoming.begin(), mvIncoming.begin() + mOptions.nMaxBatchObjects);
                        mvIncoming.erase(mvIncoming.begin(), mvIncoming.begin() + mOptions.nMaxBatchObjects);
                    }
                }
            }
            // The ones put off before come along when their turn is due -- and can
            // begin a batch by themselves, or they would wait for the next retire,
            // which after the last keyframe never comes.
            std::vector<Entry> vStill;
            for(const Entry &entry : mvPutOff)
            {
                if(entry.when <= start)
                {
                    mvBatch.push_back(entry);
                    bBegin = true;
                }
                else
                    vStill.push_back(entry);
            }
            mvPutOff.swap(vStill);
            if(!bBegin)
                return;
            mnGraceEpoch = mEpoch.fetch_add(1) + 1;
            Enter(State::GRACE1, start);
            ++delta.nBatches;
        }

        std::vector<const void*> vpPins;
        if(mState == State::GRACE1 && GraceOver(mnGraceEpoch, vpPins))
        {
            PutOffAll(vpPins, delta.nPutOffPinned, start);
            if(mOptions.bCheckAndFree)
            {
                mvpCandidates.clear();
                mvpCandidates.reserve(mvBatch.size());
                for(const Entry &entry : mvBatch)
                    mvpCandidates.push_back(entry.p);
                mCandidates.Build(mvpCandidates);
                mvpWaiting.clear();
                for(const Entry &entry : mvPutOff)
                    mvpWaiting.push_back(entry.p);
                {
                    std::lock_guard<std::mutex> lock(mMutexIncoming);
                    for(const Entry &entry : mvIncoming)
                        mvpWaiting.push_back(entry.p);
                }
                mvpNamed.clear();
                mbScanRestart = true;
                Enter(State::CHECK, start);
            }
            else
            {
                mnGraceEpoch = mEpoch.fetch_add(1) + 1;
                Enter(State::GRACE2, start);
            }
        }

        if(mState == State::CHECK)
        {
            const std::chrono::microseconds used = spent();
            if(used < budget)
            {
                const bool bDone = mvBatch.empty() ||
                                   mHooks.scan(mCandidates, mvpWaiting, mvpNamed, mbScanRestart, budget - used);
                mbScanRestart = false;
                if(bDone)
                {
                    const std::vector<const void*> vpNamed(mvpNamed.begin(), mvpNamed.end());
                    PutOffAll(vpNamed, delta.nPutOffNamed, start);
                    mvpNamed.clear();
                    mnGraceEpoch = mEpoch.fetch_add(1) + 1;
                    Enter(State::GRACE2, start);
                }
            }
        }

        if(mState == State::GRACE2 && GraceOver(mnGraceEpoch, vpPins))
        {
            // Nothing should be pinned now that was not pinned before the check: a
            // reader can only have got it from a holder the check looked in. If it
            // happens, the list of holders is incomplete. In a dry run there was no
            // check, so a pin here is an ordinary one.
            PutOffAll(vpPins, mOptions.bCheckAndFree ? delta.nLatePins : delta.nPutOffPinned, start);
            mnFreeCursor = 0;
            Enter(State::FREE, start);
        }

        if(mState == State::FREE)
        {
            while(mnFreeCursor < mvBatch.size())
            {
                if((mnFreeCursor & 63) == 0 && spent() >= budget)
                    break;
                void* const p = mvBatch[mnFreeCursor++].p;
                if(mOptions.bCheckAndFree)
                {
                    mHooks.destroy(p);
                    ++delta.nFreed;
                }
                else
                    ++delta.nKept;
            }
            if(mnFreeCursor >= mvBatch.size())
            {
                mvBatch.clear();
                mvpWaiting.clear();
                mnFreeCursor = 0;
                Enter(State::IDLE, start);
            }
        }

        const std::chrono::microseconds took = spent();
        std::size_t nIncoming = 0;
        {
            std::lock_guard<std::mutex> lock(mMutexIncoming);
            nIncoming = mvIncoming.size();
        }
        std::lock_guard<std::mutex> lock(mMutexStats);
        mStats.nFreed += delta.nFreed;
        mStats.nKept += delta.nKept;
        mStats.nBatches += delta.nBatches;
        mStats.nPutOffPinned += delta.nPutOffPinned;
        mStats.nPutOffNamed += delta.nPutOffNamed;
        mStats.nLatePins += delta.nLatePins;
        mStats.nWaiting = nIncoming + mvPutOff.size() + (mvBatch.size() - mnFreeCursor);
        mStats.nPeakWaiting = std::max(mStats.nPeakWaiting, mStats.nWaiting);
        mStats.longestStep = std::max(mStats.longestStep, took);
        mStats.state = mState;
        mStats.stateSince = mStateSince;
    }

    Reclaimer::Stats Reclaimer::GetStats() const
    {
        Stats stats;
        {
            std::lock_guard<std::mutex> lock(mMutexStats);
            stats = mStats;
        }
        std::lock_guard<std::mutex> lock(mMutexIncoming);
        stats.nRetired = mnRetired;
        stats.nWaiting = static_cast<std::size_t>(mnRetired - stats.nFreed - stats.nKept);
        return stats;
    }

} // namespace ORB_SLAM3
