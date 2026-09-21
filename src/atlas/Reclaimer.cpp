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
#include <utility>

namespace ORB_SLAM3
{

    namespace
    {
        // An object that was put off n times is looked at again every 2^n batches,
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

    Reclaimer::Reclaimer(Hooks hooks, Options options) : mHooks(std::move(hooks)), mOptions(options) {}

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
        // It holds nothing now, which is what announcing says.
        slot.nAnnounced = mEpoch.load();
        slot.nSeen = slot.nAnnounced;
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

    void Reclaimer::Step(std::chrono::microseconds budget)
    {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const auto spent = [&start] { return std::chrono::steady_clock::now() - start; };
        Stats delta;

        // Removes from the batch what `vp` names, counting it under `counter`.
        const auto putOffAll = [this](const std::vector<const void*> &vp, std::uint64_t &counter)
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
                if(entry.p && named.Contains(entry.p))
                {
                    if(entry.nPutOff < 255)
                        ++entry.nPutOff;
                    mvPutOff.push_back(entry);
                    entry.p = nullptr;
                    ++counter;
                }
            mvBatch.erase(std::remove_if(mvBatch.begin(), mvBatch.end(), [](const Entry &e) { return !e.p; }),
                          mvBatch.end());
        };
        const auto rebuildCandidates = [this]
        {
            mvpCandidates.clear();
            mvpCandidates.reserve(mvBatch.size());
            for(const Entry &entry : mvBatch)
                mvpCandidates.push_back(entry.p);
            mCandidates.Build(mvpCandidates);
        };

        bool bBegin = false;
        if(mState == State::IDLE)
        {
            std::lock_guard<std::mutex> lock(mMutexIncoming);
            const bool bFull = mvIncoming.size() >= mOptions.nBatchObjects;
            const bool bOld = !mvIncoming.empty() && start - mvIncoming.front().retired >= mOptions.maxAge;
            bBegin = bFull || bOld;
            if(bBegin)
                mvBatch.swap(mvIncoming);
        }
        if(bBegin)
        {
            // The ones put off before come along when their turn is due.
            std::vector<Entry> vStill;
            for(const Entry &entry : mvPutOff)
            {
                const unsigned shift = std::min<unsigned>(entry.nPutOff, kMaxPutOffShift);
                if(mnBatch % (1ull << shift) == 0)
                    mvBatch.push_back(entry);
                else
                    vStill.push_back(entry);
            }
            mvPutOff.swap(vStill);

            mnGraceEpoch = mEpoch.fetch_add(1) + 1;
            mState = State::GRACE1;
            ++delta.nBatches;
        }

        std::vector<const void*> vpPins;
        if(mState == State::GRACE1)
        {
            if(GraceOver(mnGraceEpoch, vpPins))
            {
                putOffAll(vpPins, delta.nPutOffPinned);
                rebuildCandidates();
                mvpNamed.clear();
                mbScanRestart = true;
                if(mOptions.bCheckAndFree)
                    mState = State::CHECK;
                else
                {
                    mnGraceEpoch = mEpoch.fetch_add(1) + 1;
                    mState = State::GRACE2;
                }
            }
        }

        if(mState == State::CHECK && spent() < budget)
        {
            const std::chrono::microseconds left = budget -
                                                   std::chrono::duration_cast<std::chrono::microseconds>(spent());
            const bool bDone = mvBatch.empty() || mHooks.scan(mCandidates, mvpNamed, mbScanRestart, left);
            mbScanRestart = false;
            if(bDone)
            {
                std::vector<const void*> vpNamed(mvpNamed.begin(), mvpNamed.end());
                putOffAll(vpNamed, delta.nPutOffNamed);
                rebuildCandidates();
                mnGraceEpoch = mEpoch.fetch_add(1) + 1;
                mState = State::GRACE2;
            }
        }

        if(mState == State::GRACE2)
        {
            if(GraceOver(mnGraceEpoch, vpPins))
            {
                // Nothing should be pinned that was not pinned before the check: a
                // reader can only have got it from a holder the check looked in. If
                // it happens, the list of holders is incomplete.
                putOffAll(vpPins, delta.nLatePins);
                mnFreeCursor = 0;
                mState = State::FREE;
            }
        }

        if(mState == State::FREE)
        {
            while(mnFreeCursor < mvBatch.size())
            {
                if((mnFreeCursor & 63) == 0 && spent() >= budget)
                    break;
                if(mOptions.bCheckAndFree)
                {
                    mHooks.destroy(mvBatch[mnFreeCursor].p);
                    ++delta.nFreed;
                }
                else
                    ++delta.nKept;
                ++mnFreeCursor;
            }
            if(mnFreeCursor >= mvBatch.size())
            {
                mvBatch.clear();
                mvpCandidates.clear();
                mCandidates.Build(mvpCandidates);
                mnFreeCursor = 0;
                ++mnBatch;
                mState = State::IDLE;
            }
        }

        const std::chrono::microseconds took = std::chrono::duration_cast<std::chrono::microseconds>(spent());
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
        mStats.nWaiting = nIncoming + mvPutOff.size() + (mvBatch.size() - std::min(mnFreeCursor, mvBatch.size()));
        mStats.nPeakWaiting = std::max(mStats.nPeakWaiting, mStats.nWaiting);
        mStats.longestStep = std::max(mStats.longestStep, took);
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
        return stats;
    }

} // namespace ORB_SLAM3
