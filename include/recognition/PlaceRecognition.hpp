/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez,
* José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
* University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the
* terms of the GNU General Public License as published by the Free Software Foundation,
* either version 3 of the License, or (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "KeyFrame.hpp"
#include "MapPoint.hpp"
#include "Atlas.hpp"
#include "KeyFrameDatabase.hpp"
#include "ORBVocabulary.hpp"
#include "ORBmatcher.hpp"
#include "Sim3Solver.hpp"
#include "Converter.hpp"
#include "Optimizer.hpp"
#include "core/SensorTypes.hpp"

#include "g2o/types/types_seven_dof_expmap.h"

#include <vector>
#include <set>
#include <mutex>

namespace ORB_SLAM3 {

class KeyFrame;
class MapPoint;
class Atlas;
class KeyFrameDatabase;

// =============================================================================
// PlaceRecognition -- Detects common regions (loop closures and map merges)
// Extracted from LoopClosing per ORB-SLAM3 paper architecture (Fig. 1)
// =============================================================================
class PlaceRecognition {
public:

    // ---- Output struct returned by detect() --------------------------------
    struct DetectionResult {
        bool loopDetected = false;
        bool mergeDetected = false;

        // Loop closure data
        KeyFrame* loopMatchedKF = nullptr;
        KeyFrame* loopLastCurrentKF = nullptr;
        g2o::Sim3 loopSlw;
        g2o::Sim3 loopScw;
        std::vector<MapPoint*> loopMPs;
        std::vector<MapPoint*> loopMatchedMPs;

        // Map merge data
        KeyFrame* mergeMatchedKF = nullptr;
        KeyFrame* mergeLastCurrentKF = nullptr;
        g2o::Sim3 mergeSlw;
        g2o::Sim3 mergeSmw;
        g2o::Sim3 mergeScw;
        std::vector<MapPoint*> mergeMPs;
        std::vector<MapPoint*> mergeMatchedMPs;
        std::vector<KeyFrame*> mergeConnectedKFs;
    };

    // ---- Construction ------------------------------------------------------
    PlaceRecognition(Atlas* pAtlas, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,
                     bool bFixScale, bool bActiveLC);

    // ---- Main entry point --------------------------------------------------
    // Runs the NewDetectCommonRegions logic on pCurrentKF.
    // The sensor type is needed for stereo / IMU-monocular checks that the
    // original code performed through mpTracker->mSensor.
    DetectionResult detect(KeyFrame* pCurrentKF, eSensor sensor);

    // ---- State resets (called by LoopClosing after correction) -------------
    void resetLoopState();
    void resetMergeState();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    // ---- Helper methods (ported from LoopClosing) --------------------------
    bool DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF,
                                        g2o::Sim3 &gScw, int &nNumProjMatches,
                                        std::vector<MapPoint*> &vpMPs,
                                        std::vector<MapPoint*> &vpMatchedMPs,
                                        eSensor sensor);

    bool DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand,
                                    KeyFrame* &pMatchedKF, KeyFrame* &pLastCurrentKF,
                                    g2o::Sim3 &g2oScw, int &nNumCoincidences,
                                    std::vector<MapPoint*> &vpMPs,
                                    std::vector<MapPoint*> &vpMatchedMPs,
                                    KeyFrame* pCurrentKF,
                                    eSensor sensor);

    bool DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF,
                                       g2o::Sim3 &gScw, int &nNumProjMatches,
                                       std::vector<MapPoint*> &vpMPs,
                                       std::vector<MapPoint*> &vpMatchedMPs);

    int FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw,
                                g2o::Sim3 &g2oScw,
                                std::set<MapPoint*> &spMatchedMPinOrigin,
                                std::vector<MapPoint*> &vpMapPoints,
                                std::vector<MapPoint*> &vpMatchedMapPoints);

    // ---- Core references ---------------------------------------------------
    Atlas* mpAtlas;
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    bool mbFixScale;
    bool mbActiveLC;

    // ---- Loop detection state ----------------------------------------------
    bool mbLoopDetected = false;
    int mnLoopNumCoincidences = 0;
    int mnLoopNumNotFound = 0;
    KeyFrame* mpLoopLastCurrentKF = nullptr;
    KeyFrame* mpLoopMatchedKF = nullptr;
    g2o::Sim3 mg2oLoopSlw;
    g2o::Sim3 mg2oLoopScw;
    std::vector<MapPoint*> mvpLoopMPs;
    std::vector<MapPoint*> mvpLoopMatchedMPs;

    // ---- Merge detection state ---------------------------------------------
    bool mbMergeDetected = false;
    int mnMergeNumCoincidences = 0;
    int mnMergeNumNotFound = 0;
    KeyFrame* mpMergeLastCurrentKF = nullptr;
    KeyFrame* mpMergeMatchedKF = nullptr;
    g2o::Sim3 mg2oMergeSlw;
    g2o::Sim3 mg2oMergeSmw;
    g2o::Sim3 mg2oMergeScw;
    std::vector<MapPoint*> mvpMergeMPs;
    std::vector<MapPoint*> mvpMergeMatchedMPs;
    std::vector<KeyFrame*> mvpMergeConnectedKFs;
};

} // namespace ORB_SLAM3
