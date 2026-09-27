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

#ifndef SLOTPOOL_H
#define SLOTPOOL_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ORB_SLAM3
{

    // Fixed-size slots in slabs, for objects that are made and freed by the
    // hundred thousand and walked by the thousand every frame: MapPoints.
    //
    // glibc served them well enough while nothing was ever freed -- every new
    // point went to the top of the heap, next to the last one. Once culled
    // points are freed, glibc hands the holes out again, and a frame's new
    // points land wherever old ones died, across hundreds of megabytes; walking
    // them cost tracking 3 % on KITTI stereo (docs/OWNERSHIP.md, step 6). Here
    // consecutive allocations fill one slab: a fresh slab from the front, a
    // used one hole by hole, and when the slab is full the one with the most
    // room comes next. Freed slots go back to their slab; a slab with nothing
    // left in it goes back to the system. Freeing never reaches glibc's bins,
    // so the rest of the heap is laid out as if nothing had been freed.
    //
    // A slab is kSlabBytes, aligned to kSlabBytes, with its bookkeeping at the
    // front, so a slot's slab is its address rounded down. One mutex; a few
    // hundred calls a frame.
    class SlotPool
    {
    public:
        static constexpr std::size_t kSlabBytes = 128 * 1024;

        explicit SlotPool(std::size_t nSlotBytes);
        ~SlotPool();

        SlotPool(const SlotPool &) = delete;
        SlotPool &operator=(const SlotPool &) = delete;

        void* Allocate();
        void Free(void* p);

        struct Stats
        {
            std::size_t nSlabs = 0, nSlots = 0, nUsed = 0, nBytes = 0;
        };
        Stats GetStats() const;

        std::size_t SlotBytes() const { return mnSlotBytes; }

    private:
        struct Slab
        {
            std::size_t nIndex;  // in mvSlabs
            std::size_t nFree;   // free slots, whether never used or on the list
            std::size_t nBumped; // slots handed out from the front so far
            void* pFreeList;     // freed slots, linked through their first word
        };
        static Slab* SlabOf(void* p);
        char* SlotAt(Slab* pSlab, std::size_t i) const;
        Slab* NewSlab();
        void ReleaseSlab(Slab* pSlab);

        const std::size_t mnSlotBytes;
        const std::size_t mnSlotsPerSlab;
        mutable std::mutex mMutex;
        std::vector<Slab*> mvSlabs;
        Slab* mpCurrent = nullptr;
        std::size_t mnUsed = 0;
    };

} // namespace ORB_SLAM3

#endif // SLOTPOOL_H
