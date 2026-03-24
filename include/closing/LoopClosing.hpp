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


#pragma once

#include "KeyFrame.hpp"
#include "Atlas.hpp"
#include "ORBVocabulary.hpp"
#include "core/Types.hpp"
#include "core/Interfaces.hpp"
#include "core/SensorTypes.hpp"

#include "KeyFrameDatabase.hpp"

#include <boost/algorithm/string.hpp>
#include <thread>
#include <mutex>

namespace ORB_SLAM3
{

class KeyFrameDatabase;
class Map;
class PlaceRecognition;


class LoopClosing : public IKeyFrameConsumer
{
public:

public:

    LoopClosing(Atlas* pAtlas, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,
                const bool bFixScale, const bool bActiveLC, int sensor);

    void SetTracker(ITrackingState* pTracker);
    void SetLocalMapper(IMappingControl* pLocalMapper);
    void SetPlaceRecognition(PlaceRecognition* pPR) { mpPlaceRecognition = pPR; }

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF) override;

    void RequestReset();
    void RequestResetActiveMap(Map* pMap);

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF);

    bool isRunningGBA(){
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

#ifdef REGISTER_TIMES

    std::vector<double> vdDataQuery_ms;
    std::vector<double> vdEstSim3_ms;
    std::vector<double> vdPRTotal_ms;

    std::vector<double> vdMergeMaps_ms;
    std::vector<double> vdWeldingBA_ms;
    std::vector<double> vdMergeOptEss_ms;
    std::vector<double> vdMergeTotal_ms;
    std::vector<int> vnMergeKFs;
    std::vector<int> vnMergeMPs;
    int nMerges;

    std::vector<double> vdLoopFusion_ms;
    std::vector<double> vdLoopOptEss_ms;
    std::vector<double> vdLoopTotal_ms;
    std::vector<int> vnLoopKFs;
    int nLoop;

    std::vector<double> vdGBA_ms;
    std::vector<double> vdUpdateMap_ms;
    std::vector<double> vdFGBATotal_ms;
    std::vector<int> vnGBAKFs;
    std::vector<int> vnGBAMPs;
    int nFGBA_exec;
    int nFGBA_abort;

#endif

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool CheckNewKeyFrames();


    // Place recognition delegated to PlaceRecognition module

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, std::vector<MapPoint*> &vpMapPoints);
    void SearchAndFuse(const std::vector<KeyFrame*> &vConectedKFs, std::vector<MapPoint*> &vpMapPoints);

    void CorrectLoop();

    void MergeLocal();
    void MergeLocal2();

    void CheckObservations(std::set<KeyFrame*> &spKFsMap1, std::set<KeyFrame*> &spKFsMap2);

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetActiveMapRequested;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;
    ITrackingState* mpTracker;  // Only for UpdateFrameIMU/GetLastKeyFrame
    int mSensor;                // Sensor type (avoids mpTracker->mSensor lookups)

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    IMappingControl *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    KeyFrame* mpLastCurrentKF;
    KeyFrame* mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    std::vector<MapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;
    Sim3Type mg2oScw;

    //-------
    Map* mpLastMap;
    PlaceRecognition* mpPlaceRecognition = nullptr;

    bool mbLoopDetected;
    int mnLoopNumCoincidences;
    int mnLoopNumNotFound;
    KeyFrame* mpLoopLastCurrentKF;
    Sim3Type mg2oLoopSlw;
    Sim3Type mg2oLoopScw;
    KeyFrame* mpLoopMatchedKF;
    std::vector<MapPoint*> mvpLoopMPs;
    std::vector<MapPoint*> mvpLoopMatchedMPs;
    bool mbMergeDetected;
    int mnMergeNumCoincidences;
    int mnMergeNumNotFound;
    KeyFrame* mpMergeLastCurrentKF;
    Sim3Type mg2oMergeSlw;
    Sim3Type mg2oMergeSmw;
    Sim3Type mg2oMergeScw;
    KeyFrame* mpMergeMatchedKF;
    std::vector<MapPoint*> mvpMergeMPs;
    std::vector<MapPoint*> mvpMergeMatchedMPs;
    std::vector<KeyFrame*> mvpMergeConnectedKFs;

    Sim3Type mSold_new;
    //-------

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;


    int mnFullBAIdx;



    std::vector<double> vdPR_CurrentTime;
    std::vector<double> vdPR_MatchedTime;
    std::vector<int> vnPR_TypeRecogn;

    //DEBUG
    std::string mstrFolderSubTraj;
    int mnNumCorrection;
    int mnCorrectionGBA;


    // To (de)activate LC
    bool mbActiveLC = true;

#ifdef REGISTER_LOOP
    std::string mstrFolderLoop;
#endif
};

} //namespace ORB_SLAM
