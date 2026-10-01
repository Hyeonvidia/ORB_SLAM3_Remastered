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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "common/ThreadPorts.hpp"
#include "common/FlatMap.hpp"
#include "features/ORBdescriptor.hpp"
#include "common/Converter.hpp"
#include "atlas/SlotPool.hpp"

#include "common/SerializationUtils.hpp"

#include <memory>
#include <cstring>
#include <atomic>
#include <opencv2/core/core.hpp>
#include <cstdint>
#include <mutex>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

#include <map>
#include <set>
#include <tuple>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    class Frame;

    class MapPoint
    {
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            // clang-format off
            ar & mnId;
            ar & mnFirstKFid;
            ar & mnFirstFrame;
            ar & nObs;
            // Variables used by the tracking
            //ar & mTrackProjX;
            //ar & mTrackProjY;
            //ar & mTrackDepth;
            //ar & mTrackDepthR;
            //ar & mTrackProjXR;
            //ar & mTrackProjYR;
            //ar & mbTrackInView;
            //ar & mbTrackInViewR;
            //ar & mnTrackScaleLevel;
            //ar & mnTrackScaleLevelR;
            //ar & mTrackViewCos;
            //ar & mTrackViewCosR;
            //ar & mnTrackReferenceForFrame;
            //ar & mnLastFrameSeen;

            // Variables used by local mapping
            //ar & mnBALocalForKF;
            //ar & mnFuseCandidateForKF;

            // Variables used by loop closing and merging
            //ar & mnLoopPointForKF;
            //ar & mnCorrectedByKF;
            //ar & mnCorrectedReference;
            //serializeMatrix(ar,mPosGBA,version);
            //ar & mnBAGlobalForKF;
            //ar & mnBALocalForMerge;
            //serializeMatrix(ar,mPosMerge,version);
            //serializeMatrix(ar,mNormalVectorMerge,version);

            // Protected variables
            ar & boost::serialization::make_array(mWorldPos.data(), mWorldPos.size());
            ar & boost::serialization::make_array(mNormalVector.data(), mNormalVector.size());
            //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
            //ar & mObservations;
            if(!mpBackup)
                mpBackup = std::make_unique<Backup>();
            ar & mpBackup->mObservationsId1;
            ar & mpBackup->mObservationsId2;
            {
                cv::Mat descriptor(1, 32, CV_8U, mDescriptorData);
                serializeMatrix(ar, descriptor, version);
                if(descriptor.data != mDescriptorData && descriptor.total() == 32)
                    std::memcpy(mDescriptorData, descriptor.data, 32);
            }
            ar & mpBackup->mRefKFId;
            //ar & mnVisible;
            //ar & mnFound;

            ar & mbBad;
            ar & mpBackup->mReplacedId;

            ar & mfMinDistance;
            ar & mfMaxDistance;

            // clang-format on
        }

    public:
        // The keyframes that observe the point, and where in each: the index
        // of the feature in the left image and in the right, or -1.
        typedef FlatMap<KeyFrame*, std::tuple<int, int>> ObservationMap;

        // From a pool of MapPoints' own (atlas/SlotPool.hpp), so that a frame's
        // new points sit together whatever has been freed, and freeing them
        // never reaches glibc. 16-byte aligned, as Eigen's operator new was.
        static void* operator new(std::size_t nBytes);
        static void operator delete(void* p);
        static void operator delete(void* p, std::size_t nBytes);
        static const SlotPool &Pool();

        MapPoint();
        ~MapPoint();

        // What this point holds, by member group; see MemoryAudit.
        std::map<std::string, std::size_t> MemoryFootprint() const;

        MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap);
        MapPoint(const Eigen::Vector3f &Pos, Map* pMap, Frame* pFrame, const int &idxF);

        void SetWorldPos(const Eigen::Vector3f &Pos);
        Eigen::Vector3f GetWorldPos();

        Eigen::Vector3f GetNormal();
        void SetNormalVector(const Eigen::Vector3f &normal);

        KeyFrame* GetReferenceKeyFrame();

        ObservationMap GetObservations();
        int Observations();

        void AddObservation(KeyFrame* pKF, int idx);
        void EraseObservation(KeyFrame* pKF);

        std::tuple<int, int> GetIndexInKeyFrame(KeyFrame* pKF);
        bool IsInKeyFrame(KeyFrame* pKF);

        void SetBadFlag();
        bool isBad();

        void Replace(MapPoint* pMP);
        MapPoint* GetReplaced();

        void IncreaseVisible(int n = 1);
        void IncreaseFound(int n = 1);
        float GetFoundRatio();
        inline int GetFound() { return mnFound; }

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();
        ORBdescriptor::Bytes GetDescriptorBytes();

        void UpdateNormalAndDepth();

        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();
        int PredictScale(const float &currentDist, KeyFrame* pKF);
        int PredictScale(const float &currentDist, Frame* pF);

        Map* GetMap();
        void UpdateMap(Map* pMap);

        void PrintObservations();

        void PreSave(std::set<KeyFrame*> &spKF, std::set<MapPoint*> &spMP);
        void PostLoad(std::map<long unsigned int, KeyFrame*> &mpKFid, std::map<long unsigned int, MapPoint*> &mpMPid);

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        // Variables used by the tracking
        float mTrackProjX;
        float mTrackProjY;
        float mTrackDepth;
        float mTrackDepthR;
        float mTrackProjXR;
        float mTrackProjYR;
        bool mbTrackInView, mbTrackInViewR;
        int mnTrackScaleLevel, mnTrackScaleLevelR;
        float mTrackViewCos, mTrackViewCosR;
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;

        // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopPointForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        Eigen::Vector3f mPosGBA;
        long unsigned int mnBAGlobalForKF;
        long unsigned int mnBALocalForMerge;

        // Variable used by merging
        Eigen::Vector3f mPosMerge;
        Eigen::Vector3f mNormalVectorMerge;

        // Fopr inverse depth optimization

        static std::mutex mGlobalMutex;

        unsigned int mnOriginMapId;

    protected:
        // Position in absolute coordinates
        Eigen::Vector3f mWorldPos;

        // Keyframes observing the point and associated index in keyframe
        ObservationMap mObservations;
        // For save relation without pointer, this is necessary for save/load function
        // What a saved atlas holds instead of pointers; made for saving and
        // loading, and nothing between, so that a point does not carry it.
        struct Backup
        {
            std::map<long unsigned int, int> mObservationsId1;
            std::map<long unsigned int, int> mObservationsId2;
            long unsigned int mRefKFId = 0;
            long long int mReplacedId = -1;
        };
        std::unique_ptr<Backup> mpBackup;

        // Mean viewing direction
        Eigen::Vector3f mNormalVector;

        // Best descriptor to fast matching. The 32 bytes live in the object;
        // mDescriptor is a header over them, so a point is one allocation, its
        // descriptor sits next to its position, and freeing a point frees nothing
        // else. copyTo() into a header of the same shape reuses the buffer.
        // The 32 bytes themselves: a cv::Mat around them is 96 bytes of header
        // for every point, and nothing reads one.
        alignas(16) std::uint8_t mDescriptorData[32] = {};

        // Reference KeyFrame
        KeyFrame* mpRefKF;

        // Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        MapPoint* mpReplaced;
        // For save relation without pointer, this is necessary for save/load function

        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;

        // Which map the point is in: moved by merges, read by every thread.
        SharedPointer<Map> mpMap;

        // Mutex
        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };

} // namespace ORB_SLAM3

#endif // MAPPOINT_H
