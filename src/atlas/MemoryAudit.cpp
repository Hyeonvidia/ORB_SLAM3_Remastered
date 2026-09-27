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

#include "atlas/MemoryAudit.hpp"

#include "atlas/KeyFrame.hpp"
#include "atlas/MapPoint.hpp"

#include <malloc.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <ostream>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_set>

namespace ORB_SLAM3
{

    namespace
    {
        std::mutex gMutex;
        std::unordered_set<const KeyFrame*> gKeyFrames;
        std::unordered_set<const MapPoint*> gMapPoints;

        void Add(MemoryAudit::Footprint &total, const MemoryAudit::Footprint &part)
        {
            for(const auto &entry : part)
                total[entry.first] += entry.second;
        }

        void Print(std::ostream &os, const char* title, std::size_t count, const MemoryAudit::Footprint &f)
        {
            std::size_t sum = 0;
            for(const auto &entry : f)
                sum += entry.second;
            char line[160];
            std::snprintf(line, sizeof(line), "  %-34s %9zu objects %10.1f MB %8.1f KB each\n", title, count,
                          sum / 1048576.0, count ? sum / 1024.0 / count : 0.0);
            os << line;
            for(const auto &entry : f)
            {
                std::snprintf(line, sizeof(line), "      %-38s %10.1f MB %8.2f KB each\n", entry.first.c_str(),
                              entry.second / 1048576.0, count ? entry.second / 1024.0 / count : 0.0);
                os << line;
            }
        }

        long StatusKB(const char* key)
        {
            std::ifstream f("/proc/self/status");
            std::string word;
            while(f >> word)
                if(word == key)
                {
                    long kb = 0;
                    f >> kb;
                    return kb;
                }
            return 0;
        }

        // Who still names the objects that were culled. The threads have stopped
        // when this runs, so it is what a collector would find at the end of the
        // run: a culled object somebody still names cannot be freed by one that
        // only proves unreachability, and one nobody names can.
        //
        // Each culled object is counted once, under the first holder below that
        // names it, because that is the order a collector can do something about
        // them: a live keyframe's slot stays until that keyframe is culled; a
        // culled keyframe's slots go when its payload is released; a replaced-by
        // link goes with the point that holds it.
        void Census(std::ostream &os, const std::unordered_set<const KeyFrame*> &keyFrames,
                    const std::unordered_set<const MapPoint*> &mapPoints)
        {
            std::unordered_set<const MapPoint*> inLiveSlot, inDeadSlot, replacedBy;
            std::unordered_set<const KeyFrame*> observed, reference, covisible, inTree, inEdge, inImuChain;
            std::size_t nLiveSlotsHoldingDead = 0, nObservedNotInSlot = 0;

            for(const KeyFrame* pConstKF : keyFrames)
            {
                KeyFrame* pKF = const_cast<KeyFrame*>(pConstKF);
                const bool bDeadKF = pKF->isBad();
                for(MapPoint* pMP : pKF->GetMapPointMatches())
                {
                    if(!pMP || !mapPoints.count(pMP) || !pMP->isBad())
                        continue;
                    (bDeadKF ? inDeadSlot : inLiveSlot).insert(pMP);
                    if(!bDeadKF)
                        ++nLiveSlotsHoldingDead;
                }
                if(bDeadKF)
                    continue;
                for(KeyFrame* pOther : pKF->GetVectorCovisibleKeyFrames())
                    if(pOther && pOther->isBad())
                        covisible.insert(pOther);
                for(KeyFrame* pOther : pKF->GetChilds())
                    if(pOther && pOther->isBad())
                        inTree.insert(pOther);
                if(pKF->GetParent() && pKF->GetParent()->isBad())
                    inTree.insert(pKF->GetParent());
                for(KeyFrame* pOther : pKF->GetLoopEdges())
                    if(pOther && pOther->isBad())
                        inEdge.insert(pOther);
                for(KeyFrame* pOther : pKF->GetMergeEdges())
                    if(pOther && pOther->isBad())
                        inEdge.insert(pOther);
                if(pKF->mPrevKF && pKF->mPrevKF->isBad())
                    inImuChain.insert(pKF->mPrevKF);
                if(pKF->mNextKF && pKF->mNextKF->isBad())
                    inImuChain.insert(pKF->mNextKF);
            }
            for(const MapPoint* pConstMP : mapPoints)
            {
                MapPoint* pMP = const_cast<MapPoint*>(pConstMP);
                MapPoint* pRep = pMP->GetReplaced();
                if(pRep && mapPoints.count(pRep) && pRep->isBad())
                    replacedBy.insert(pRep);
                if(pMP->isBad())
                    continue;
                for(const auto &obs : pMP->GetObservations())
                    if(obs.first && obs.first->isBad())
                    {
                        observed.insert(obs.first);
                        // Releasing a culled keyframe's slots and keypoints rests on its
                        // slots naming every live point that observes it.
                        const int idx = std::get<0>(obs.second) != -1 ? std::get<0>(obs.second)
                                                                      : std::get<1>(obs.second);
                        const std::vector<MapPoint*> vpSlots = obs.first->GetMapPointMatches();
                        if(idx < 0 || idx >= static_cast<int>(vpSlots.size()) || vpSlots[idx] != pMP)
                            ++nObservedNotInSlot;
                    }
                if(pMP->GetReferenceKeyFrame() && pMP->GetReferenceKeyFrame()->isBad())
                    reference.insert(pMP->GetReferenceKeyFrame());
            }

            std::size_t nDeadMP = 0, nLive = 0, nDead = 0, nReplaced = 0, nFree = 0;
            std::size_t bLive = 0, bDead = 0, bReplaced = 0, bFree = 0;
            for(const MapPoint* pMP : mapPoints)
            {
                if(!const_cast<MapPoint*>(pMP)->isBad())
                    continue;
                ++nDeadMP;
                std::size_t bytes = 0;
                for(const auto &entry : pMP->MemoryFootprint())
                    bytes += entry.second;
                if(inLiveSlot.count(pMP))
                    ++nLive, bLive += bytes;
                else if(inDeadSlot.count(pMP))
                    ++nDead, bDead += bytes;
                else if(replacedBy.count(pMP))
                    ++nReplaced, bReplaced += bytes;
                else
                    ++nFree, bFree += bytes;
            }

            std::size_t nDeadKF = 0, nObserved = 0, nReference = 0, nCovisible = 0, nTree = 0, nEdge = 0, nImu = 0,
                        nNobody = 0;
            std::size_t bObserved = 0, bReference = 0, bCovisible = 0, bTree = 0, bEdge = 0, bImu = 0, bNobody = 0;
            for(const KeyFrame* pKF : keyFrames)
            {
                if(!const_cast<KeyFrame*>(pKF)->isBad())
                    continue;
                ++nDeadKF;
                std::size_t bytes = 0;
                for(const auto &entry : pKF->MemoryFootprint())
                    bytes += entry.second;
                if(observed.count(pKF))
                    ++nObserved, bObserved += bytes;
                else if(reference.count(pKF))
                    ++nReference, bReference += bytes;
                else if(covisible.count(pKF))
                    ++nCovisible, bCovisible += bytes;
                else if(inTree.count(pKF))
                    ++nTree, bTree += bytes;
                else if(inEdge.count(pKF))
                    ++nEdge, bEdge += bytes;
                else if(inImuChain.count(pKF))
                    ++nImu, bImu += bytes;
                else
                    ++nNobody, bNobody += bytes;
            }

            char line[200];
            const auto row = [&](const char* what, std::size_t n, std::size_t total, double mb)
            {
                std::snprintf(line, sizeof(line), "      %-52s %9zu %5.1f %% %9.1f MB\n", what, n,
                              total ? 100.0 * n / total : 0.0, mb);
                os << line;
            };
            os << "  Who still names the culled MapPoints (each counted once, first holder that applies)\n";
            row("a slot of a keyframe still in a map", nLive, nDeadMP, bLive / 1048576.0);
            row("only slots of culled keyframes", nDead, nDeadMP, bDead / 1048576.0);
            row("only another point's replaced-by link", nReplaced, nDeadMP, bReplaced / 1048576.0);
            row("no keyframe and no point", nFree, nDeadMP, bFree / 1048576.0);
            std::snprintf(line, sizeof(line), "      (%zu slots of live keyframes hold a culled point)\n",
                          nLiveSlotsHoldingDead);
            os << line;
            os << "  Who still names the culled KeyFrames (each counted once, first holder that applies)\n";
            row("a live point's observations", nObserved, nDeadKF, bObserved / 1048576.0);
            row("only a live point's reference keyframe", nReference, nDeadKF, bReference / 1048576.0);
            row("a live keyframe's covisibility list", nCovisible, nDeadKF, bCovisible / 1048576.0);
            row("a live keyframe's parent or children", nTree, nDeadKF, bTree / 1048576.0);
            row("a live keyframe's loop or merge edges", nEdge, nDeadKF, bEdge / 1048576.0);
            row("a live keyframe's previous/next (IMU chain)", nImu, nDeadKF, bImu / 1048576.0);
            row("none of these", nNobody, nDeadKF, bNobody / 1048576.0);
            std::snprintf(
                line, sizeof(line),
                "      (%zu observations of a culled keyframe by a live point that is not in that keyframe's slot)\n",
                nObservedNotInSlot);
            os << line;
        }
    } // namespace

    bool MemoryAudit::SlotAuditEnabled()
    {
        static const bool bEnabled = []
        {
            const char* env = std::getenv("ORBSLAM3R_SLOT_AUDIT");
            return Enabled() && env && std::string(env) == "1";
        }();
        return bEnabled;
    }

    void MemoryAudit::AuditSlots(const char* where)
    {
        if(!SlotAuditEnabled())
            return;
        static std::set<std::pair<const MapPoint*, const KeyFrame*>> sReported;
        static std::size_t nReported = 0;
        std::vector<const MapPoint*> vpPoints;
        {
            std::lock_guard<std::mutex> lock(gMutex);
            vpPoints.assign(gMapPoints.begin(), gMapPoints.end());
        }
        std::size_t nNew = 0;
        for(const MapPoint* pConst : vpPoints)
        {
            MapPoint* pMP = const_cast<MapPoint*>(pConst);
            if(pMP->isBad())
                continue;
            for(const auto &obs : pMP->GetObservations())
            {
                KeyFrame* pKF = obs.first;
                const int idx = std::get<0>(obs.second) != -1 ? std::get<0>(obs.second) : std::get<1>(obs.second);
                MapPoint* pInSlot = idx >= 0 && idx < pKF->N ? pKF->GetMapPoint(idx) : nullptr;
                if(pInSlot == pMP)
                    continue;
                if(!sReported.insert(std::make_pair(pConst, static_cast<const KeyFrame*>(pKF))).second)
                    continue;
                ++nNew;
                if(++nReported <= 40)
                    std::fprintf(stderr,
                                 "SLOT AUDIT after %s: point %lu (first KF %ld) observes keyframe %lu (%s) at %d; the "
                                 "slot holds %s%lu\n",
                                 where, pMP->mnId, pMP->mnFirstKFid, pKF->mnId, pKF->isBad() ? "culled" : "live", idx,
                                 pInSlot ? (pInSlot->isBad() ? "culled point " : "point ") : "nothing ",
                                 pInSlot ? pInSlot->mnId : 0ul);
            }
        }
        if(nNew)
            std::fprintf(stderr, "SLOT AUDIT after %s: %zu new mismatched observations, %zu so far\n", where, nNew,
                         nReported);
    }

    bool MemoryAudit::Enabled()
    {
        static const bool bEnabled = []
        {
            const char* env = std::getenv("ORBSLAM3R_MEMORY_REPORT");
            return env && std::string(env) == "1";
        }();
        return bEnabled;
    }

    void MemoryAudit::Register(const KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lock(gMutex);
        gKeyFrames.insert(pKF);
    }

    void MemoryAudit::Unregister(const KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lock(gMutex);
        gKeyFrames.erase(pKF);
    }

    void MemoryAudit::Register(const MapPoint* pMP)
    {
        std::lock_guard<std::mutex> lock(gMutex);
        gMapPoints.insert(pMP);
    }

    void MemoryAudit::Unregister(const MapPoint* pMP)
    {
        std::lock_guard<std::mutex> lock(gMutex);
        gMapPoints.erase(pMP);
    }

    void MemoryAudit::Report(std::ostream &os, const Footprint &vocabulary)
    {
        std::lock_guard<std::mutex> lock(gMutex);

        Footprint kfLive, kfDead, mpLive, mpDead;
        std::size_t nKFLive = 0, nKFDead = 0, nMPLive = 0, nMPDead = 0;
        for(const KeyFrame* pKF : gKeyFrames)
        {
            const bool bDead = const_cast<KeyFrame*>(pKF)->isBad();
            Add(bDead ? kfDead : kfLive, pKF->MemoryFootprint());
            ++(bDead ? nKFDead : nKFLive);
        }
        for(const MapPoint* pMP : gMapPoints)
        {
            const bool bDead = const_cast<MapPoint*>(pMP)->isBad();
            Add(bDead ? mpDead : mpLive, pMP->MemoryFootprint());
            ++(bDead ? nMPDead : nMPLive);
        }

        os << "\n== MEMORY REPORT ==\n";
        Print(os, "KeyFrames in a map", nKFLive, kfLive);
        Print(os, "KeyFrames culled, never freed", nKFDead, kfDead);
        Print(os, "MapPoints in a map", nMPLive, mpLive);
        Print(os, "MapPoints culled, never freed", nMPDead, mpDead);
        Print(os, "Vocabulary", 1, vocabulary);
        Census(os, gKeyFrames, gMapPoints);

        const struct mallinfo2 mi = mallinfo2();
        char line[200];
        std::snprintf(
            line, sizeof(line),
            "  process: RSS %ld MB, peak RSS %ld MB; heap in use %.0f MB, free in arenas %.0f MB, mmapped %.0f MB\n",
            StatusKB("VmRSS:") / 1024, StatusKB("VmHWM:") / 1024, mi.uordblks / 1048576.0, mi.fordblks / 1048576.0,
            mi.hblkhd / 1048576.0);
        os << line << "== END MEMORY REPORT ==\n";
    }

} // namespace ORB_SLAM3
