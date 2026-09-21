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
    } // namespace

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
