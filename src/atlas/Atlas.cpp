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

#include "atlas/Atlas.hpp"
#include "atlas/Reclaimer.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#endif

#include "camera/GeometricCamera.hpp"
#include "camera/Pinhole.hpp"
#include "camera/KannalaBrandt8.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <vector>
#include "atlas/KeyFrame.hpp"
#include "atlas/Map.hpp"
#include "atlas/MapPoint.hpp"
#include "atlas/ORBVocabulary.hpp"

namespace ORB_SLAM3
{

    namespace
    {
        // The check: where a MapPoint* can still be, looked in rather than
        // reasoned about (docs/OWNERSHIP.md). Runs on the Reclaimer's driver, in
        // slices; the keyframe list is taken once per batch and walked with a
        // cursor.
        struct HolderScan
        {
            Atlas* pAtlas;
            std::vector<KeyFrame*> vpKeyFrames;
            std::vector<MapPoint*> vpSlots;
            std::size_t nCursor = 0, nWaitingCursor = 0;

            bool Run(const Reclaimer::PointerSet &candidates, const std::vector<void*> &waiting,
                     std::vector<void*> &vpNamed, bool bRestart, std::chrono::microseconds budget)
            {
                const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                if(bRestart)
                {
                    vpKeyFrames.clear();
                    nCursor = 0;
                    nWaitingCursor = 0;
                    for(Map* pMap : pAtlas->GetAllMaps())
                    {
                        // The live keyframes and the culled ones: a culled keyframe
                        // keeps its slots, and is still read through the members
                        // that name it.
                        for(KeyFrame* pKF : pMap->GetAllKeyFrames())
                            vpKeyFrames.push_back(pKF);
                        for(KeyFrame* pKF : pMap->GetCulledKeyFrames())
                            vpKeyFrames.push_back(pKF);
                        // What the viewer draws, one frame behind Tracking, and for
                        // as long as the map is not the current one.
                        for(MapPoint* pMP : pMap->GetReferenceMapPoints())
                            if(candidates.Contains(pMP))
                                vpNamed.push_back(pMP);
                    }
                }
                // The budget is looked at every few keyframes and every few
                // hundred links: a keyframe is 2000 slots copied under its mutex
                // and probed, a link is two mutexes.
                for(; nCursor < vpKeyFrames.size(); ++nCursor)
                {
                    if((nCursor & 3) == 0 && std::chrono::steady_clock::now() - start > budget)
                        return false;
                    vpKeyFrames[nCursor]->CopyMapPointMatches(vpSlots);
                    for(MapPoint* pMP : vpSlots)
                        if(candidates.Contains(pMP))
                            vpNamed.push_back(pMP);
                }
                // A replaced point names its replacement, and Tracking follows
                // that link from a point it still holds. From every retired point
                // still waiting, and then from every named candidate, to a fixed
                // point: a chain of replacements can lie inside one batch.
                for(; nWaitingCursor < waiting.size(); ++nWaitingCursor)
                {
                    if((nWaitingCursor & 255) == 0 && std::chrono::steady_clock::now() - start > budget)
                        return false;
                    MapPoint* pRep = static_cast<MapPoint*>(waiting[nWaitingCursor])->GetReplaced();
                    if(candidates.Contains(pRep))
                        vpNamed.push_back(pRep);
                }
                std::unordered_set<void*> named(vpNamed.begin(), vpNamed.end());
                for(std::size_t i = 0; i < vpNamed.size(); ++i)
                {
                    MapPoint* pRep = static_cast<MapPoint*>(vpNamed[i])->GetReplaced();
                    if(candidates.Contains(pRep) && named.insert(pRep).second)
                        vpNamed.push_back(pRep);
                }
                return true;
            }
        };

        void Destroy(void* p)
        {
            MapPoint* pMP = static_cast<MapPoint*>(p);
            if(Atlas::ReclaimMode() == Atlas::Reclaim::POISON)
            {
                // Its members go, its memory stays: nothing can be allocated over
                // it, so a later touch cannot be masked by a new object there.
                pMP->~MapPoint();
                std::memset(static_cast<void*>(pMP), 0xDD, sizeof(MapPoint));
#if defined(__SANITIZE_ADDRESS__)
                ASAN_POISON_MEMORY_REGION(pMP, sizeof(MapPoint));
#endif
                return;
            }
            delete pMP;
        }

        std::unique_ptr<Reclaimer> MakeReclaimer(Atlas* pAtlas)
        {
            const Atlas::Reclaim mode = Atlas::ReclaimMode();
            Reclaimer::Hooks hooks;
            Reclaimer::Options options;
            options.bCheckAndFree = mode == Atlas::Reclaim::POINTS || mode == Atlas::Reclaim::POISON;
            if(options.bCheckAndFree)
            {
                std::shared_ptr<HolderScan> pScan(new HolderScan{pAtlas});
                hooks.scan = [pScan](const Reclaimer::PointerSet &candidates, const std::vector<void*> &waiting,
                                     std::vector<void*> &vpNamed, bool bRestart, std::chrono::microseconds budget)
                { return pScan->Run(candidates, waiting, vpNamed, bRestart, budget); };
                hooks.destroy = &Destroy;
            }
            if(const char* env = std::getenv("ORBSLAM3R_RECLAIM_BATCH"))
                options.nBatchObjects = static_cast<std::size_t>(std::max(1L, std::atol(env)));
            return std::unique_ptr<Reclaimer>(new Reclaimer(hooks, options));
        }
    } // namespace

    Atlas::Reclaim Atlas::ReclaimMode()
    {
        static const Reclaim mode = []
        {
            const char* env = std::getenv("ORBSLAM3R_RECLAIM");
            const std::string s = env ? env : "";
            if(s == "count")
                return Reclaim::COUNT;
            if(s == "points")
                return Reclaim::POINTS;
            if(s == "poison")
                return Reclaim::POISON;
            return Reclaim::DRY;
        }();
        return mode;
    }

    Atlas::Atlas() : mpReclaimer(MakeReclaimer(this))
    {
        mpCurrentMap = static_cast<Map*>(NULL);
    }

    Atlas::Atlas(int initKFid) : mpReclaimer(MakeReclaimer(this)), mnLastInitKFidMap(initKFid)
    {
        mpCurrentMap = static_cast<Map*>(NULL);
        CreateNewMap();
    }

    Atlas::~Atlas()
    {
        for(std::set<Map*>::iterator it = mspMaps.begin(), end = mspMaps.end(); it != end;)
        {
            Map* pMi = *it;

            if(pMi)
            {
                delete pMi;
                pMi = static_cast<Map*>(NULL);

                it = mspMaps.erase(it);
            }
            else
                ++it;
        }
    }

    void Atlas::CreateNewMap()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        std::cout << "Creation of new map with id: " << Map::nNextId << std::endl;
        if(mpCurrentMap)
        {
            if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
                mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() + 1; //The init KF is the next of current maximum

            mpCurrentMap->SetStoredMap();
            std::cout << "Stored map with ID: " << mpCurrentMap->GetId() << std::endl;
        }
        std::cout << "Creation of new map with last KF id: " << mnLastInitKFidMap << std::endl;

        mpCurrentMap = new Map(mnLastInitKFidMap);
        mpCurrentMap->SetReclaimer(mpReclaimer.get());
        mpCurrentMap->SetCurrentMap();
        mspMaps.insert(mpCurrentMap);
    }

    void Atlas::ChangeMap(Map* pMap)
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        std::cout << "Change to map with id: " << pMap->GetId() << std::endl;
        if(mpCurrentMap)
        {
            mpCurrentMap->SetStoredMap();
        }

        mpCurrentMap = pMap;
        mpCurrentMap->SetCurrentMap();
    }

    unsigned long int Atlas::GetLastInitKFid()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mnLastInitKFidMap;
    }

    void Atlas::AddKeyFrame(KeyFrame* pKF)
    {
        Map* pMapKF = pKF->GetMap();
        pMapKF->AddKeyFrame(pKF);
    }

    void Atlas::AddMapPoint(MapPoint* pMP)
    {
        Map* pMapMP = pMP->GetMap();
        pMapMP->AddMapPoint(pMP);
    }

    GeometricCamera* Atlas::AddCamera(GeometricCamera* pCam)
    {
        //Check if the camera already exists
        bool bAlreadyInMap = false;
        int index_cam = -1;
        for(size_t i = 0; i < mvpCameras.size(); ++i)
        {
            GeometricCamera* pCam_i = mvpCameras[i];
            if(!pCam)
                std::cout << "Not pCam" << std::endl;
            if(!pCam_i)
                std::cout << "Not pCam_i" << std::endl;
            if(pCam->GetType() != pCam_i->GetType())
                continue;

            if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
            {
                if(((Pinhole*)pCam_i)->IsEqual(pCam))
                {
                    bAlreadyInMap = true;
                    index_cam = i;
                }
            }
            else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
            {
                if(((KannalaBrandt8*)pCam_i)->IsEqual(pCam))
                {
                    bAlreadyInMap = true;
                    index_cam = i;
                }
            }
        }

        if(bAlreadyInMap)
        {
            return mvpCameras[index_cam];
        }
        else
        {
            mvpCameras.push_back(pCam);
            return pCam;
        }
    }

    std::vector<GeometricCamera*> Atlas::GetAllCameras()
    {
        return mvpCameras;
    }

    void Atlas::SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs)
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        mpCurrentMap->SetReferenceMapPoints(vpMPs);
    }

    void Atlas::InformNewBigChange()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        mpCurrentMap->InformNewBigChange();
    }

    int Atlas::GetLastBigChangeIdx()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->GetLastBigChangeIdx();
    }

    long unsigned int Atlas::MapPointsInMap()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->MapPointsInMap();
    }

    long unsigned Atlas::KeyFramesInMap()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->KeyFramesInMap();
    }

    std::vector<KeyFrame*> Atlas::GetAllKeyFrames()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->GetAllKeyFrames();
    }

    std::vector<MapPoint*> Atlas::GetAllMapPoints()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->GetAllMapPoints();
    }

    std::vector<MapPoint*> Atlas::GetReferenceMapPoints()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->GetReferenceMapPoints();
    }

    std::vector<Map*> Atlas::GetAllMaps()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        struct compFunctor
        {
            inline bool operator()(Map* elem1, Map* elem2) { return elem1->GetId() < elem2->GetId(); }
        };
        std::vector<Map*> vMaps(mspMaps.begin(), mspMaps.end());
        std::sort(vMaps.begin(), vMaps.end(), compFunctor());
        return vMaps;
    }

    int Atlas::CountMaps()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mspMaps.size();
    }

    void Atlas::clearMap()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        mpCurrentMap->clear();
    }

    void Atlas::clearAtlas()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        /*for(std::set<Map*>::iterator it=mspMaps.begin(), send=mspMaps.end(); it!=send; it++)
    {
        (*it)->clear();
        delete *it;
    }*/
        mspMaps.clear();
        mpCurrentMap = static_cast<Map*>(NULL);
        mnLastInitKFidMap = 0;
    }

    Map* Atlas::GetCurrentMap()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        if(!mpCurrentMap)
            CreateNewMap();
        while(mpCurrentMap->IsBad())
            usleep(3000);

        return mpCurrentMap;
    }

    void Atlas::SetMapBad(Map* pMap)
    {
        mspMaps.erase(pMap);
        pMap->SetBad();

        mspBadMaps.insert(pMap);
    }

    void Atlas::RemoveBadMaps()
    {
        /*for(Map* pMap : mspBadMaps)
    {
        delete pMap;
        pMap = static_cast<Map*>(NULL);
    }*/
        mspBadMaps.clear();
    }

    bool Atlas::isInertial()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->IsInertial();
    }

    void Atlas::SetInertialSensor()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        mpCurrentMap->SetInertialSensor();
    }

    void Atlas::SetImuInitialized()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        mpCurrentMap->SetImuInitialized();
    }

    bool Atlas::isImuInitialized()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap->isImuInitialized();
    }

    void Atlas::PreSave()
    {
        if(mpCurrentMap)
        {
            if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
                mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() + 1; //The init KF is the next of current maximum
        }

        struct compFunctor
        {
            inline bool operator()(Map* elem1, Map* elem2) { return elem1->GetId() < elem2->GetId(); }
        };
        std::copy(mspMaps.begin(), mspMaps.end(), std::back_inserter(mvpBackupMaps));
        std::sort(mvpBackupMaps.begin(), mvpBackupMaps.end(), compFunctor());

        std::set<GeometricCamera*> spCams(mvpCameras.begin(), mvpCameras.end());
        for(Map* pMi : mvpBackupMaps)
        {
            if(!pMi || pMi->IsBad())
                continue;

            if(pMi->GetAllKeyFrames().size() == 0)
            {
                // Empty map, erase before of save it.
                SetMapBad(pMi);
                continue;
            }
            pMi->PreSave(spCams);
        }
        RemoveBadMaps();
    }

    void Atlas::PostLoad()
    {
        std::map<unsigned int, GeometricCamera*> mpCams;
        for(GeometricCamera* pCam : mvpCameras)
        {
            mpCams[pCam->GetId()] = pCam;
        }

        mspMaps.clear();
        unsigned long int numKF = 0, numMP = 0;
        for(Map* pMi : mvpBackupMaps)
        {
            mspMaps.insert(pMi);
            pMi->SetReclaimer(mpReclaimer.get());
            pMi->PostLoad(mpKeyFrameDB, mpORBVocabulary, mpCams);
            numKF += pMi->GetAllKeyFrames().size();
            numMP += pMi->GetAllMapPoints().size();
        }
        mvpBackupMaps.clear();
    }

    void Atlas::ReportReclaimer(std::ostream &os)
    {
        const Reclaimer::Stats stats = mpReclaimer->GetStats();
        std::size_t nCulledKeyFrames = 0;
        {
            std::lock_guard<std::mutex> lock(mMutexAtlas);
            for(Map* pMap : mspMaps)
                nCulledKeyFrames += pMap->GetCulledKeyFrames().size();
        }
        os << "  Culled keyframes on the maps' lists: " << nCulledKeyFrames << "\n";
        const SlotPool::Stats pool = MapPoint::Pool().GetStats();
        os << "  MapPoint pool: " << pool.nSlabs << " slabs, " << pool.nUsed << " of " << pool.nSlots
           << " slots in use, " << pool.nBytes / 1048576.0 << " MB\n";
        const char* mode = ReclaimMode() == Reclaim::COUNT    ? "count"
                           : ReclaimMode() == Reclaim::DRY    ? "dry run"
                           : ReclaimMode() == Reclaim::POINTS ? "points"
                                                              : "poison";
        os << "  Reclaimer (" << mode << "): retired " << stats.nRetired << ", freed " << stats.nFreed << ", kept "
           << stats.nKept << ", waiting " << stats.nWaiting << " (peak " << stats.nPeakWaiting << "), batches "
           << stats.nBatches << ", put off " << stats.nPutOffPinned << " pinned + " << stats.nPutOffNamed
           << " named, late pins " << stats.nLatePins << ", longest step " << stats.longestStep.count() << " us (begin "
           << stats.longestPart[0].count() << ", grace 1 " << stats.longestPart[1].count() << ", check "
           << stats.longestPart[2].count() << ", grace 2 " << stats.longestPart[3].count() << ", free "
           << stats.longestPart[4].count() << ")\n";
    }

    void Atlas::SetKeyFrameDababase(KeyFrameDatabase* pKFDB)
    {
        mpKeyFrameDB = pKFDB;
    }

    KeyFrameDatabase* Atlas::GetKeyFrameDatabase()
    {
        return mpKeyFrameDB;
    }

    void Atlas::SetORBVocabulary(ORBVocabulary* pORBVoc)
    {
        mpORBVocabulary = pORBVoc;
    }

    ORBVocabulary* Atlas::GetORBVocabulary()
    {
        return mpORBVocabulary;
    }

    long unsigned int Atlas::GetNumLivedKF()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        long unsigned int num = 0;
        for(Map* pMap_i : mspMaps)
        {
            num += pMap_i->GetAllKeyFrames().size();
        }

        return num;
    }

    long unsigned int Atlas::GetNumLivedMP()
    {
        std::lock_guard<std::mutex> lock(mMutexAtlas);
        long unsigned int num = 0;
        for(Map* pMap_i : mspMaps)
        {
            num += pMap_i->GetAllMapPoints().size();
        }

        return num;
    }

    std::map<long unsigned int, KeyFrame*> Atlas::GetAtlasKeyframes()
    {
        std::map<long unsigned int, KeyFrame*> mpIdKFs;
        for(Map* pMap_i : mvpBackupMaps)
        {
            std::vector<KeyFrame*> vpKFs_Mi = pMap_i->GetAllKeyFrames();

            for(KeyFrame* pKF_j_Mi : vpKFs_Mi)
            {
                mpIdKFs[pKF_j_Mi->mnId] = pKF_j_Mi;
            }
        }

        return mpIdKFs;
    }

} //namespace ORB_SLAM3
