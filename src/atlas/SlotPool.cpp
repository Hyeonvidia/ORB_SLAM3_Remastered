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

#include "atlas/SlotPool.hpp"

#include <cassert>
#include <cstdlib>
#include <new>

namespace ORB_SLAM3
{

    namespace
    {
        // The bookkeeping at the front of a slab, rounded so that the first slot
        // is as aligned as any.
        constexpr std::size_t kHeaderBytes = 64;
    } // namespace

    SlotPool::SlotPool(std::size_t nSlotBytes)
        : mnSlotBytes((nSlotBytes + 15) / 16 * 16), mnSlotsPerSlab((kSlabBytes - kHeaderBytes) / mnSlotBytes)
    {
        static_assert(sizeof(Slab) <= kHeaderBytes, "the slab's bookkeeping must fit its header");
        assert(mnSlotsPerSlab > 0);
    }

    SlotPool::~SlotPool()
    {
        // Whatever is still allocated belongs to objects that were never freed,
        // as before there was a pool; the slabs go with the process.
    }

    SlotPool::Slab* SlotPool::SlabOf(void* p)
    {
        return reinterpret_cast<Slab*>(reinterpret_cast<std::uintptr_t>(p) & ~(std::uintptr_t(kSlabBytes) - 1));
    }

    char* SlotPool::SlotAt(Slab* pSlab, std::size_t i) const
    {
        return reinterpret_cast<char*>(pSlab) + kHeaderBytes + i * mnSlotBytes;
    }

    SlotPool::Slab* SlotPool::NewSlab()
    {
        void* pMemory = std::aligned_alloc(kSlabBytes, kSlabBytes);
        if(!pMemory)
            throw std::bad_alloc();
        Slab* pSlab = static_cast<Slab*>(pMemory);
        pSlab->nIndex = mvSlabs.size();
        pSlab->nFree = mnSlotsPerSlab;
        pSlab->nBumped = 0;
        pSlab->pFreeList = nullptr;
        mvSlabs.push_back(pSlab);
        return pSlab;
    }

    void SlotPool::ReleaseSlab(Slab* pSlab)
    {
        Slab* pLast = mvSlabs.back();
        mvSlabs[pSlab->nIndex] = pLast;
        pLast->nIndex = pSlab->nIndex;
        mvSlabs.pop_back();
        std::free(pSlab);
    }

    void* SlotPool::Allocate()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if(!mpCurrent || mpCurrent->nFree == 0)
        {
            // The slab with the most room, so that the next allocations stay
            // together for as long as possible; a new one only when none has any.
            mpCurrent = nullptr;
            for(Slab* pSlab : mvSlabs)
                if(pSlab->nFree > 0 && (!mpCurrent || pSlab->nFree > mpCurrent->nFree))
                    mpCurrent = pSlab;
            if(!mpCurrent)
                mpCurrent = NewSlab();
        }
        Slab* pSlab = mpCurrent;
        void* p;
        if(pSlab->pFreeList)
        {
            p = pSlab->pFreeList;
            pSlab->pFreeList = *static_cast<void**>(p);
        }
        else
        {
            p = SlotAt(pSlab, pSlab->nBumped++);
        }
        --pSlab->nFree;
        ++mnUsed;
        return p;
    }

    void SlotPool::Free(void* p)
    {
        if(!p)
            return;
        std::lock_guard<std::mutex> lock(mMutex);
        Slab* pSlab = SlabOf(p);
        *static_cast<void**>(p) = pSlab->pFreeList;
        pSlab->pFreeList = p;
        ++pSlab->nFree;
        --mnUsed;
        // Empty, and not the one being filled: back to the system. The one being
        // filled stays, or a run of make-and-free would make and free slabs.
        if(pSlab->nFree == mnSlotsPerSlab && pSlab != mpCurrent && mvSlabs.size() > 1)
            ReleaseSlab(pSlab);
    }

    SlotPool::Stats SlotPool::GetStats() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        Stats stats;
        stats.nSlabs = mvSlabs.size();
        stats.nSlots = mvSlabs.size() * mnSlotsPerSlab;
        stats.nUsed = mnUsed;
        stats.nBytes = mvSlabs.size() * kSlabBytes;
        return stats;
    }

} // namespace ORB_SLAM3
