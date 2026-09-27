// atlas/SlotPool: slots come out aligned and distinct, a run of allocations
// after a batch of frees stays inside one slab, empty slabs go back, the
// counts add up, and four threads can allocate and free at once.
#include "atlas/SlotPool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <thread>
#include <vector>

namespace
{
    int gFailures = 0;
    void Expect(bool ok, const char* what)
    {
        if(!ok)
        {
            std::fprintf(stderr, "FAILED: %s\n", what);
            ++gFailures;
        }
    }
    std::uintptr_t SlabOf(void* p)
    {
        return reinterpret_cast<std::uintptr_t>(p) & ~(std::uintptr_t(ORB_SLAM3::SlotPool::kSlabBytes) - 1);
    }
} // namespace

int main()
{
    using ORB_SLAM3::SlotPool;
    const std::size_t kObject = 700; // about a MapPoint
    SlotPool pool(kObject);
    Expect(pool.SlotBytes() % 16 == 0 && pool.SlotBytes() >= kObject,
           "slots are 16-byte multiples that fit the object");

    // Fresh allocations: distinct, aligned, and consecutive within a slab.
    std::vector<void*> vp;
    for(int i = 0; i < 1000; ++i)
        vp.push_back(pool.Allocate());
    std::set<void*> distinct(vp.begin(), vp.end());
    Expect(distinct.size() == vp.size(), "every slot is distinct");
    Expect(std::all_of(vp.begin(), vp.end(), [](void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }),
           "every slot is 16-byte aligned");
    std::size_t nSameSlabAsPrevious = 0;
    for(std::size_t i = 1; i < vp.size(); ++i)
        nSameSlabAsPrevious += SlabOf(vp[i]) == SlabOf(vp[i - 1]);
    Expect(nSameSlabAsPrevious >= vp.size() - 8, "fresh allocations fill one slab after another");
    for(void* p : vp)
        std::memset(p, 0xAB, kObject); // the whole slot is ours

    SlotPool::Stats s = pool.GetStats();
    Expect(s.nUsed == 1000 && s.nSlots >= 1000 && s.nSlots - s.nUsed < s.nSlots / 5 + 200,
           "stats: 1000 in use, little waste");
    const std::size_t nSlotsPerSlab = s.nSlots / s.nSlabs;

    // Free every other slot of the first slabs, then allocate again: the new
    // ones must land in those holes, one slab at a time.
    std::vector<void*> vpKept;
    for(std::size_t i = 0; i < vp.size(); ++i)
        if(i % 2 == 0)
            pool.Free(vp[i]);
        else
            vpKept.push_back(vp[i]);
    std::vector<void*> vpAgain;
    for(int i = 0; i < 300; ++i)
        vpAgain.push_back(pool.Allocate());
    std::set<std::uintptr_t> slabsUsed;
    for(void* p : vpAgain)
        slabsUsed.insert(SlabOf(p));
    Expect(slabsUsed.size() <= (300 + nSlotsPerSlab / 2 - 1) / (nSlotsPerSlab / 2) + 1,
           "allocations after frees fill one slab's holes before moving to the next");
    Expect(pool.GetStats().nSlabs == s.nSlabs, "no new slab while old ones have room");
    for(void* p : vpAgain)
        Expect(std::count(vpKept.begin(), vpKept.end(), p) == 0, "a reused slot is not one still in use");

    // Free everything: every slab but the current one goes back.
    for(void* p : vpKept)
        pool.Free(p);
    for(void* p : vpAgain)
        pool.Free(p);
    s = pool.GetStats();
    Expect(s.nUsed == 0, "nothing in use after freeing everything");
    Expect(s.nSlabs <= 2, "empty slabs are released");

    // Four threads allocating and freeing at once; every slot handed out is
    // distinct from every other slot in use at the time.
    std::atomic<int> nErrors{0};
    std::vector<std::thread> vThreads;
    for(int t = 0; t < 4; ++t)
        vThreads.emplace_back(
            [&pool, &nErrors, t]
            {
                std::mt19937 rng(t + 1);
                std::vector<void*> mine;
                for(int i = 0; i < 20000; ++i)
                {
                    if(mine.empty() || rng() % 3 != 0)
                    {
                        void* p = pool.Allocate();
                        // Write our thread's mark; any other thread holding the same
                        // slot would overwrite it.
                        std::memset(p, 0x10 + t, 64);
                        mine.push_back(p);
                    }
                    else
                    {
                        const std::size_t k = rng() % mine.size();
                        if(static_cast<unsigned char*>(mine[k])[0] != 0x10 + t ||
                           static_cast<unsigned char*>(mine[k])[63] != 0x10 + t)
                            ++nErrors;
                        pool.Free(mine[k]);
                        mine[k] = mine.back();
                        mine.pop_back();
                    }
                }
                for(void* p : mine)
                    pool.Free(p);
            });
    for(std::thread &th : vThreads)
        th.join();
    Expect(nErrors == 0, "no slot was handed to two threads at once");
    Expect(pool.GetStats().nUsed == 0, "everything the threads took, they gave back");

    std::printf("%s\n", gFailures == 0 ? "PASS" : "FAIL");
    return gFailures == 0 ? 0 : 1;
}
