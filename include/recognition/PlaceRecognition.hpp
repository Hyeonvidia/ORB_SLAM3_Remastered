#pragma once

#include "KeyFrame.hpp"
#include "MapPoint.hpp"
#include "Atlas.hpp"
#include "KeyFrameDatabase.hpp"
#include "ORBVocabulary.hpp"
#include "Settings.hpp"

#include "g2o/types/types_seven_dof_expmap.h"

#include <vector>
#include <set>
#include <mutex>

namespace ORB_SLAM3 {

class KeyFrame;

// =============================================================================
// PlaceRecognition — Detects common regions across maps
// Extracted from LoopClosing per paper architecture (Fig. 1)
// =============================================================================
class PlaceRecognition {
public:
    PlaceRecognition(Atlas* pAtlas, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,
                     bool bFixScale, bool bActiveLC);

    // Main detection — returns true if common region found
    bool detectCommonRegions(KeyFrame* pCurrentKF,
                             KeyFrame*& pMatchedKF,
                             KeyFrame*& pLastCurrentKF,
                             g2o::Sim3& gScw,
                             int& nNumProjMatches,
                             std::vector<MapPoint*>& vpMPs,
                             std::vector<MapPoint*>& vpMatchedMPs,
                             bool& bLoopDetected,
                             bool& bMergeDetected);

protected:
    bool DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF,
                                         g2o::Sim3 &gScw, int &nNumProjMatches,
                                         std::vector<MapPoint*> &vpMPs,
                                         std::vector<MapPoint*> &vpMatchedMPs);

    bool DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand,
                                     KeyFrame* &pMatchedKF, KeyFrame* &pLastCurrentKF,
                                     g2o::Sim3 &g2oScw, int &nNumCoincidences,
                                     std::vector<MapPoint*> &vpMPs,
                                     std::vector<MapPoint*> &vpMatchedMPs);

    bool DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF,
                                        g2o::Sim3 &gScw, int &nNumProjMatches,
                                        std::vector<MapPoint*> &vpMPs,
                                        std::vector<MapPoint*> &vpMatchedMPs);

    int FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw,
                                 g2o::Sim3 &g2oScw,
                                 std::set<MapPoint*> &spMatchedMPinOrigin,
                                 std::vector<MapPoint*> &vpMapPoints,
                                 std::vector<MapPoint*> &vpMatchedMapPoints);

    Atlas* mpAtlas;
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    bool mbFixScale;
    bool mbActiveLC;

    // State from previous detection
    KeyFrame* mpLastCurrentKF = nullptr;
    KeyFrame* mpMatchedKF = nullptr;
    bool mbLoopDetected = false;
    bool mbMergeDetected = false;
    g2o::Sim3 mg2oLoopSlw;
    g2o::Sim3 mg2oMergeSlw;
    g2o::Sim3 mg2oMergeScw;

    // Consecutive detection tracking
    int mnLoopNumCoincidences = 0;
    int mnMergeNumCoincidences = 0;
    std::vector<MapPoint*> mvpLoopMPs;
    std::vector<MapPoint*> mvpMergeMPs;
    std::vector<MapPoint*> mvpLoopMatchedMPs;
    std::vector<MapPoint*> mvpMergeMatchedMPs;
};

} // namespace ORB_SLAM3
