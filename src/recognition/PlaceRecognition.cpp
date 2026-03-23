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

#include "PlaceRecognition.hpp"

#include "Sim3Solver.hpp"
#include "Converter.hpp"
#include "Optimizer.hpp"
#include "ORBmatcher.hpp"

#include <mutex>

namespace ORB_SLAM3
{

// =============================================================================
// Constructor
// =============================================================================
PlaceRecognition::PlaceRecognition(Atlas* pAtlas, KeyFrameDatabase* pDB,
                                   ORBVocabulary* pVoc, bool bFixScale, bool bActiveLC)
    : mpAtlas(pAtlas), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc),
      mbFixScale(bFixScale), mbActiveLC(bActiveLC),
      mbLoopDetected(false), mnLoopNumCoincidences(0), mnLoopNumNotFound(0),
      mpLoopLastCurrentKF(nullptr), mpLoopMatchedKF(nullptr),
      mbMergeDetected(false), mnMergeNumCoincidences(0), mnMergeNumNotFound(0),
      mpMergeLastCurrentKF(nullptr), mpMergeMatchedKF(nullptr)
{
}

// =============================================================================
// resetLoopState / resetMergeState
// =============================================================================
void PlaceRecognition::resetLoopState()
{
    if(mpLoopLastCurrentKF)
        mpLoopLastCurrentKF->SetErase();
    if(mpLoopMatchedKF)
        mpLoopMatchedKF->SetErase();
    mnLoopNumCoincidences = 0;
    mvpLoopMatchedMPs.clear();
    mvpLoopMPs.clear();
    mnLoopNumNotFound = 0;
    mbLoopDetected = false;
    mpLoopLastCurrentKF = nullptr;
    mpLoopMatchedKF = nullptr;
}

void PlaceRecognition::resetMergeState()
{
    if(mpMergeLastCurrentKF)
        mpMergeLastCurrentKF->SetErase();
    if(mpMergeMatchedKF)
        mpMergeMatchedKF->SetErase();
    mnMergeNumCoincidences = 0;
    mvpMergeMatchedMPs.clear();
    mvpMergeMPs.clear();
    mnMergeNumNotFound = 0;
    mbMergeDetected = false;
    mpMergeLastCurrentKF = nullptr;
    mpMergeMatchedKF = nullptr;
    mvpMergeConnectedKFs.clear();
}

// =============================================================================
// detect()  --  ported from LoopClosing::NewDetectCommonRegions()
// =============================================================================
PlaceRecognition::DetectionResult PlaceRecognition::detect(KeyFrame* pCurrentKF, System::eSensor sensor)
{
    DetectionResult result;

    // To deactivate place recognition. No loop closing nor merging will be performed.
    if(!mbActiveLC)
        return result;

    // Avoid that a keyframe can be erased while it is being processed
    pCurrentKF->SetNotErase();
    pCurrentKF->mbCurrentPlaceRecognition = true;

    Map* pLastMap = pCurrentKF->GetMap();

    if(pLastMap->IsInertial() && !pLastMap->GetIniertialBA2())
    {
        mpKeyFrameDB->add(pCurrentKF);
        pCurrentKF->SetErase();
        return result;
    }

    if(sensor == System::STEREO && pLastMap->GetAllKeyFrames().size() < 5)
    {
        mpKeyFrameDB->add(pCurrentKF);
        pCurrentKF->SetErase();
        return result;
    }

    if(pLastMap->GetAllKeyFrames().size() < 12)
    {
        mpKeyFrameDB->add(pCurrentKF);
        pCurrentKF->SetErase();
        return result;
    }

    //Check the last candidates with geometric validation
    // Loop candidates
    bool bLoopDetectedInKF = false;
    bool bCheckSpatial = false;

    if(mnLoopNumCoincidences > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (pCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse()).cast<double>();
        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw;
        int numProjMatches = 0;
        std::vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(pCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs, sensor);
        if(bCommonRegion)
        {
            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = pCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;

            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if(!mbLoopDetected)
            {
                std::cout << "PR: Loop detected with Reffine Sim3" << std::endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;

            mnLoopNumNotFound++;
            if(mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }
        }
    }

    // Merge candidates
    bool bMergeDetectedInKF = false;
    if(mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (pCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse()).cast<double>();

        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3 gScw = gScl * mg2oMergeSlw;
        int numProjMatches = 0;
        std::vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(pCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs, sensor);
        if(bCommonRegion)
        {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = pCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        }
        else
        {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if(mnMergeNumNotFound >= 2)
            {
                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }
        }
    }

    if(mbMergeDetected || mbLoopDetected)
    {
        // Populate result and return early
        mpKeyFrameDB->add(pCurrentKF);

        if(mbLoopDetected)
        {
            result.loopDetected = true;
            result.loopMatchedKF = mpLoopMatchedKF;
            result.loopLastCurrentKF = mpLoopLastCurrentKF;
            result.loopSlw = mg2oLoopSlw;
            result.loopScw = mg2oLoopSlw; // loopScw assigned from loopSlw (see LoopClosing::Run)
            result.loopMPs = mvpLoopMPs;
            result.loopMatchedMPs = mvpLoopMatchedMPs;
        }
        if(mbMergeDetected)
        {
            result.mergeDetected = true;
            result.mergeMatchedKF = mpMergeMatchedKF;
            result.mergeLastCurrentKF = mpMergeLastCurrentKF;
            result.mergeSlw = mg2oMergeSlw;
            result.mergeSmw = mg2oMergeSmw;
            result.mergeScw = mg2oMergeSlw; // mergeScw assigned from mergeSlw (see LoopClosing::Run)
            result.mergeMPs = mvpMergeMPs;
            result.mergeMatchedMPs = mvpMergeMatchedMPs;
            result.mergeConnectedKFs = mvpMergeConnectedKFs;
        }
        return result;
    }

    // Extract candidates from the bag of words
    const std::vector<KeyFrame*> vpConnectedKeyFrames = pCurrentKF->GetVectorCovisibleKeyFrames();

    std::vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    if(!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
        mpKeyFrameDB->DetectNBestCandidates(pCurrentKF, vpLoopBowCand, vpMergeBowCand, 3);
    }

    // Check the BoW candidates if the geometric candidate list is empty
    // Loop candidates
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF,
                                                     mg2oLoopSlw, mnLoopNumCoincidences,
                                                     mvpLoopMPs, mvpLoopMatchedMPs,
                                                     pCurrentKF, sensor);
    }
    // Merge candidates
    if(!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF,
                                                      mg2oMergeSlw, mnMergeNumCoincidences,
                                                      mvpMergeMPs, mvpMergeMatchedMPs,
                                                      pCurrentKF, sensor);
    }

    mpKeyFrameDB->add(pCurrentKF);

    if(mbMergeDetected || mbLoopDetected)
    {
        if(mbLoopDetected)
        {
            result.loopDetected = true;
            result.loopMatchedKF = mpLoopMatchedKF;
            result.loopLastCurrentKF = mpLoopLastCurrentKF;
            result.loopSlw = mg2oLoopSlw;
            result.loopScw = mg2oLoopSlw;
            result.loopMPs = mvpLoopMPs;
            result.loopMatchedMPs = mvpLoopMatchedMPs;
        }
        if(mbMergeDetected)
        {
            result.mergeDetected = true;
            result.mergeMatchedKF = mpMergeMatchedKF;
            result.mergeLastCurrentKF = mpMergeLastCurrentKF;
            result.mergeSlw = mg2oMergeSlw;
            result.mergeSmw = mg2oMergeSmw;
            result.mergeScw = mg2oMergeSlw;
            result.mergeMPs = mvpMergeMPs;
            result.mergeMatchedMPs = mvpMergeMatchedMPs;
            result.mergeConnectedKFs = mvpMergeConnectedKFs;
        }
        return result;
    }

    pCurrentKF->SetErase();
    pCurrentKF->mbCurrentPlaceRecognition = false;

    return result;
}

// =============================================================================
// DetectAndReffineSim3FromLastKF  --  ported from LoopClosing (lines 535-576)
// =============================================================================
bool PlaceRecognition::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF,
                                                       g2o::Sim3 &gScw, int &nNumProjMatches,
                                                       std::vector<MapPoint*> &vpMPs,
                                                       std::vector<MapPoint*> &vpMatchedMPs,
                                                       System::eSensor sensor)
{
    std::set<MapPoint*> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if(nNumProjMatches >= nProjMatches)
    {
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::Sim3 gSwm(mTwm.unit_quaternion(), mTwm.translation(), 1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale;
        if(sensor == System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale = false;
        int numOptMatches = Optimizer::OptimizeSim3(pCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);

        if(numOptMatches > nProjOptMatches)
        {
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(), 1.0);

            std::vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

// =============================================================================
// DetectCommonRegionsFromBoW  --  ported from LoopClosing (lines 578-896)
// =============================================================================
bool PlaceRecognition::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand,
                                                   KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF,
                                                   g2o::Sim3 &g2oScw, int &nNumCoincidences,
                                                   std::vector<MapPoint*> &vpMPs,
                                                   std::vector<MapPoint*> &vpMatchedMPs,
                                                   KeyFrame* pCurrentKF,
                                                   System::eSensor sensor)
{
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    std::set<KeyFrame*> spConnectedKeyFrames = pCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 10;

    ORBmatcher matcherBoW(0.9, true);
    ORBmatcher matcher(0.75, true);

    // Variables to select the best number
    KeyFrame* pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    std::vector<int> vnStage(numCandidates, 0);
    std::vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    for(KeyFrame* pKFi : vpBowCand)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        // Current KF against KF with covisibles version
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if(vpCovKFi.empty())
        {
            std::cout << "Covisible list empty" << std::endl;
            vpCovKFi.push_back(pKFi);
        }
        else
        {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }

        bool bAbortByNearKF = false;
        for(int j = 0; j < (int)vpCovKFi.size(); ++j)
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                break;
            }
        }
        if(bAbortByNearKF)
        {
            continue;
        }

        std::vector<std::vector<MapPoint*>> vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(pCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(nullptr));

        int nIndexMostBoWMatchesKF = 0;
        for(int j = 0; j < (int)vpCovKFi.size(); ++j)
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            int num = matcherBoW.SearchByBoW(pCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            if(num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        for(int j = 0; j < (int)vpCovKFi.size(); ++j)
        {
            for(int k = 0; k < (int)vvpMatchedMPs[j].size(); ++k)
            {
                MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                if(!pMPi_j || pMPi_j->isBad())
                    continue;

                if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++;

                    vpMatchedPoints[k] = pMPi_j;
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j];
                }
            }
        }

        if(numBoWMatches >= nBoWMatches)
        {
            // Geometric validation
            bool bFixedScale = mbFixScale;
            if(sensor == System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
                bFixedScale = false;

            Sim3Solver solver = Sim3Solver(pCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers

            bool bNoMore = false;
            std::vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            while(!bConverge && !bNoMore)
            {
                mTcm = solver.iterate(20, bNoMore, vbInliers, nInliers, bConverge);
            }

            if(bConverge)
            {
                // Match by reprojection
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);
                std::set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                std::set<MapPoint*> spMapPoints;
                std::vector<MapPoint*> vpMapPoints;
                std::vector<KeyFrame*> vpKeyFrames;
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }

                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(), solver.GetEstimatedTranslation().cast<double>(), (double)solver.GetEstimatedScale());
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(), pMostBoWMatchesKF->GetTranslation().cast<double>(), 1.0);
                g2o::Sim3 gScw = gScm * gSmw; // Similarity matrix of current from the world position
                Sophus::Sim3f mScw = Converter::toSophus(gScw);

                std::vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));
                std::vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(nullptr));
                int numProjMatches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);

                if(numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every match
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale2 = mbFixScale;
                    if(sensor == System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
                        bFixedScale2 = false;

                    int numOptMatches = Optimizer::OptimizeSim3(pCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true);

                    if(numOptMatches >= nSim3Inliers)
                    {
                        g2o::Sim3 gSmw2(pMostBoWMatchesKF->GetRotation().cast<double>(), pMostBoWMatchesKF->GetTranslation().cast<double>(), 1.0);
                        g2o::Sim3 gScw2 = gScm * gSmw2; // Similarity matrix of current from the world position
                        Sophus::Sim3f mScw2 = Converter::toSophus(gScw2);

                        std::vector<MapPoint*> vpMatchedMP2;
                        vpMatchedMP2.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));
                        int numProjOptMatches = matcher.SearchByProjection(pCurrentKF, mScw2, vpMapPoints, vpMatchedMP2, 5, 1.0);

                        if(numProjOptMatches >= nProjOptMatches)
                        {
                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for(MapPoint* pMPi : vpMatchedMP2)
                            {
                                if(!pMPi || pMPi->isBad())
                                {
                                    continue;
                                }

                                std::tuple<size_t, size_t> indexes = pMPi->GetIndexInKeyFrame(pKFi);
                                int idx = std::get<0>(indexes);
                                if(idx >= 0)
                                {
                                    int coord_x = pKFi->mvKeysUn[idx].pt.x;
                                    if(coord_x < min_x)
                                    {
                                        min_x = coord_x;
                                    }
                                    if(coord_x > max_x)
                                    {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pKFi->mvKeysUn[idx].pt.y;
                                    if(coord_y < min_y)
                                    {
                                        min_y = coord_y;
                                    }
                                    if(coord_y > max_y)
                                    {
                                        max_y = coord_y;
                                    }
                                }
                            }

                            int nNumKFs = 0;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            std::vector<KeyFrame*> vpCurrentCovKFs = pCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);

                            int j = 0;
                            while(nNumKFs < 3 && j < (int)vpCurrentCovKFs.size())
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d mTjc = (pKFj->GetPose() * pCurrentKF->GetPoseInverse()).cast<double>();
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(), mTjc.translation(), 1.0);
                                g2o::Sim3 gSjw = gSjc * gScw2;
                                int numProjMatches_j = 0;
                                std::vector<MapPoint*> vpMatchedMPs_j;
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj, pMostBoWMatchesKF, gSjw, numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if(bValid)
                                {
                                    Sophus::SE3f Tc_w = pCurrentKF->GetPose();
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse();
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj;
                                    Eigen::Vector3f vector_dist = Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }

                            if(nNumKFs < 3)
                            {
                                vnStage[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            if(nBestMatchesReproj < numProjOptMatches)
                            {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw2;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP2;
                            }
                        }
                    }
                }
            }
        }
        index++;
    }

    if(nBestMatchesReproj > 0)
    {
        pLastCurrentKF = pCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    }
    else
    {
        int maxStage = -1;
        int maxMatched;
        for(int i = 0; i < (int)vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false;
}

// =============================================================================
// DetectCommonRegionsFromLastKF  --  ported from LoopClosing (lines 898-911)
// =============================================================================
bool PlaceRecognition::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF,
                                                      g2o::Sim3 &gScw, int &nNumProjMatches,
                                                      std::vector<MapPoint*> &vpMPs,
                                                      std::vector<MapPoint*> &vpMatchedMPs)
{
    std::set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches)
    {
        return true;
    }

    return false;
}

// =============================================================================
// FindMatchesByProjection  --  ported from LoopClosing (lines 913-967)
// =============================================================================
int PlaceRecognition::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw,
                                               g2o::Sim3 &g2oScw,
                                               std::set<MapPoint*> &spMatchedMPinOrigin,
                                               std::vector<MapPoint*> &vpMapPoints,
                                               std::vector<MapPoint*> &vpMatchedMapPoints)
{
    int nNumCovisibles = 10;
    std::vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    std::set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    std::set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();
    if(nInitialCov < nNumCovisibles)
    {
        for(int i = 0; i < nInitialCov; ++i)
        {
            std::vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            int j = 0;
            while(j < (int)vpKFs.size() && nInserted < nNumCovisibles)
            {
                if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
                {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    std::set<MapPoint*> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    for(KeyFrame* pKFi : vpCovKFm)
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    Sophus::Sim3f mScw = Converter::toSophus(g2oScw);
    ORBmatcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(nullptr));
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

} // namespace ORB_SLAM3
