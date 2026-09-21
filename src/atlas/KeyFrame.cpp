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

#include "atlas/KeyFrame.hpp"
#include "atlas/MemoryAudit.hpp"
#include "atlas/KeyFrameDatabase.hpp"
#include "common/Converter.hpp"
#include "common/ImuTypes.hpp"
#include <mutex>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include "camera/GeometricCamera.hpp"
#include "atlas/Map.hpp"
#include "atlas/MapPoint.hpp"
#include "tracking/Frame.hpp"

namespace ORB_SLAM3
{

    long unsigned int KeyFrame::nNextId = 0;

    KeyFrame::KeyFrame()
        : mnFrameId(0), mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
          mfGridElementWidthInv(0), mfGridElementHeightInv(0), mnTrackReferenceForFrame(0), mnFuseTargetForKF(0),
          mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0), mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0),
          mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0), fx(0), fy(0), cx(0), cy(0), invfx(0),
          invfy(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0), mbf(0), mb(0),
          mThDepth(0), N(0), mvKeys(), mvKeysUn(), mvuRight(), mvDepth(), mnScaleLevels(0), mfScaleFactor(0),
          mfLogScaleFactor(0), mvScaleFactors(0), mvLevelSigma2(0), mvInvLevelSigma2(0), mnMinX(0), mnMinY(0),
          mnMaxX(0), mnMaxY(0), mPrevKF(static_cast<KeyFrame*>(NULL)), mNextKF(static_cast<KeyFrame*>(NULL)),
          mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
          mHalfBaseline(0), mbCurrentPlaceRecognition(false), mnMergeCorrectedForKF(0), NLeft(0), NRight(0),
          mnNumberOfOpt(0)
    {
        if(MemoryAudit::Enabled())
            MemoryAudit::Register(this);
    }

    KeyFrame::~KeyFrame()
    {
        if(MemoryAudit::Enabled())
            MemoryAudit::Unregister(this);
    }

    std::map<std::string, std::size_t> KeyFrame::MemoryFootprint() const
    {
        auto Grid = [](const std::vector<std::vector<std::vector<std::size_t>>> &grid)
        {
            std::size_t n = MemoryAudit::Vector(grid);
            for(const std::vector<std::vector<std::size_t>> &column : grid)
            {
                n += MemoryAudit::Vector(column);
                for(const std::vector<std::size_t> &cell : column)
                    n += MemoryAudit::Vector(cell);
            }
            return n;
        };

        std::map<std::string, std::size_t> f;
        f["object itself"] = sizeof(KeyFrame);
        f["keypoints mvKeys"] = MemoryAudit::Vector(mvKeys);
        f["keypoints mvKeysUn"] = MemoryAudit::Vector(mvKeysUn);
        f["keypoints mvKeysRight"] = MemoryAudit::Vector(mvKeysRight);
        f["descriptors"] = MemoryAudit::Mat(mDescriptors);
        f["stereo mvuRight + mvDepth"] = MemoryAudit::Vector(mvuRight) + MemoryAudit::Vector(mvDepth);
        f["map point pointers"] = MemoryAudit::Vector(mvpMapPoints) + MemoryAudit::Vector(mvBackupMapPointsId);
        f["grid"] = Grid(mGrid);
        f["grid right"] = Grid(mGridRight);
        f["BowVector"] = MemoryAudit::Tree(mBowVec);
        std::size_t nFeatVec = MemoryAudit::Tree(mFeatVec);
        for(const auto &node : mFeatVec)
            nFeatVec += MemoryAudit::Vector(node.second);
        f["FeatureVector"] = nFeatVec;
        f["covisibility and spanning tree"] = MemoryAudit::Tree(mConnectedKeyFrameWeights) +
                                              MemoryAudit::Vector(mvpOrderedConnectedKeyFrames) +
                                              MemoryAudit::Vector(mvOrderedWeights) + MemoryAudit::Tree(mspChildrens) +
                                              MemoryAudit::Tree(mspLoopEdges) + MemoryAudit::Tree(mspMergeEdges) +
                                              MemoryAudit::Vector(mvpLoopCandKFs) +
                                              MemoryAudit::Vector(mvpMergeCandKFs);
        f["fisheye left-right matches"] = MemoryAudit::Vector(mvLeftToRightMatch) +
                                          MemoryAudit::Vector(mvRightToLeftMatch);
        f["scale tables"] = MemoryAudit::Vector(mvScaleFactors) + MemoryAudit::Vector(mvLevelSigma2) +
                            MemoryAudit::Vector(mvInvLevelSigma2);
        f["IMU preintegration"] = mpImuPreintegrated ? MemoryAudit::Chunk(sizeof(IMU::Preintegrated)) +
                                                           MemoryAudit::Chunk(mpImuPreintegrated->MeasurementBytes())
                                                     : 0;
        return f;
    }

    KeyFrame::KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB)
        : bImu(pMap->isImuInitialized()), mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS),
          mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.mfGridElementWidthInv),
          mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0), mnFuseTargetForKF(0),
          mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0), mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0),
          mnRelocWords(0), mnBAGlobalForKF(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0),
          mPlaceRecognitionScore(0), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy), mbf(F.mbf),
          mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mvuRight(F.mvuRight),
          mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()), mBowVec(F.mBowVec), mFeatVec(F.mFeatVec),
          mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor),
          mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2),
          mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK_(F.mK_), mPrevKF(NULL),
          mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated), mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints),
          mpKeyFrameDB(pKFDB), mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL),
          mDistCoef(F.mDistCoef), mbNotErase(false), mnDataset(F.mnDataset), mbToBeErased(false), mbBad(false),
          mHalfBaseline(F.mb / 2), mpMap(pMap), mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile),
          mnMergeCorrectedForKF(0), mpCamera(F.mpCamera), mpCamera2(F.mpCamera2),
          mvLeftToRightMatch(F.mvLeftToRightMatch), mvRightToLeftMatch(F.mvRightToLeftMatch),
          mTlr(F.GetRelativePoseTlr()), mvKeysRight(F.mvKeysRight), NLeft(F.Nleft), NRight(F.Nright),
          mTrl(F.GetRelativePoseTrl()), mnNumberOfOpt(0)
    {
        if(MemoryAudit::Enabled())
            MemoryAudit::Register(this);
        mnId = nNextId++;

        mGrid.resize(mnGridCols);
        if(F.Nleft != -1)
            mGridRight.resize(mnGridCols);
        for(int i = 0; i < mnGridCols; i++)
        {
            mGrid[i].resize(mnGridRows);
            if(F.Nleft != -1)
                mGridRight[i].resize(mnGridRows);
            for(int j = 0; j < mnGridRows; j++)
            {
                mGrid[i][j] = F.mGrid[i][j];
                if(F.Nleft != -1)
                {
                    mGridRight[i][j] = F.mGridRight[i][j];
                }
            }
        }

        if(F.HasVelocity())
            SetVelocity(F.GetVelocity());

        mImuBias = F.mImuBias;
        SetPose(F.GetPose());

        mnOriginMapId = pMap->GetId();
    }

    void KeyFrame::ComputeBoW()
    {
        if(mBowVec.empty() || mFeatVec.empty())
        {
            std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            // Feature vector associate features with nodes in the 4th level (from leaves up)
            // We assume the vocabulary tree has 6 levels, change the 4 otherwise
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void KeyFrame::SetPose(const Sophus::SE3f &Tcw)
    {
        std::lock_guard<std::mutex> lock(mMutexPose);

        mState.SetPose(Tcw, mImuCalib);
    }

    void KeyFrame::SetVelocity(const Eigen::Vector3f &Vw)
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        mState.SetVelocity(Vw);
    }

    Sophus::SE3f KeyFrame::GetPose()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.Tcw();
    }

    Sophus::SE3f KeyFrame::GetPoseInverse()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.Twc();
    }

    Eigen::Vector3f KeyFrame::GetCameraCenter()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.Ow();
    }

    Eigen::Vector3f KeyFrame::GetImuPosition()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.Owb();
    }

    Eigen::Matrix3f KeyFrame::GetImuRotation()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.ImuRotation(mImuCalib);
    }

    Sophus::SE3f KeyFrame::GetImuPose()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.ImuPose(mImuCalib);
    }

    Eigen::Matrix3f KeyFrame::GetRotation()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.Rcw();
    }

    Eigen::Vector3f KeyFrame::GetTranslation()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.tcw();
    }

    Eigen::Vector3f KeyFrame::GetVelocity()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.Velocity();
    }

    bool KeyFrame::isVelocitySet()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mState.HasVelocity();
    }

    void KeyFrame::AddConnection(KeyFrame* pKF, const int &weight)
    {
        {
            std::lock_guard<std::mutex> lock(mMutexConnections);
            if(!mConnectedKeyFrameWeights.count(pKF))
                mConnectedKeyFrameWeights[pKF] = weight;
            else if(mConnectedKeyFrameWeights[pKF] != weight)
                mConnectedKeyFrameWeights[pKF] = weight;
            else
                return;
        }

        UpdateBestCovisibles();
    }

    void KeyFrame::UpdateBestCovisibles()
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        std::vector<std::pair<int, KeyFrame*>> vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());
        for(std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                               mend = mConnectedKeyFrameWeights.end();
            mit != mend; mit++)
            vPairs.push_back(std::make_pair(mit->second, mit->first));

        std::sort(vPairs.begin(), vPairs.end());
        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;
        for(size_t i = 0, iend = vPairs.size(); i < iend; i++)
        {
            if(!vPairs[i].second->isBad())
            {
                lKFs.push_front(vPairs[i].second);
                lWs.push_front(vPairs[i].first);
            }
        }

        mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
    }

    std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        std::set<KeyFrame*> s;
        for(std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin();
            mit != mConnectedKeyFrameWeights.end(); mit++)
            s.insert(mit->first);
        return s;
    }

    std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;
    }

    std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        if((int)mvpOrderedConnectedKeyFrames.size() < N)
            return mvpOrderedConnectedKeyFrames;
        else
            return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                          mvpOrderedConnectedKeyFrames.begin() + N);
    }

    std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);

        if(mvpOrderedConnectedKeyFrames.empty())
        {
            return std::vector<KeyFrame*>();
        }

        std::vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                                                    KeyFrame::weightComp);

        if(it == mvOrderedWeights.end() && mvOrderedWeights.back() < w)
        {
            return std::vector<KeyFrame*>();
        }
        else
        {
            int n = it - mvOrderedWeights.begin();
            return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                          mvpOrderedConnectedKeyFrames.begin() + n);
        }
    }

    int KeyFrame::GetWeight(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
            return mConnectedKeyFrameWeights[pKF];
        else
            return 0;
    }

    int KeyFrame::GetNumberMPs()
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        int numberMPs = 0;
        for(size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
        {
            if(!mvpMapPoints[i])
                continue;
            numberMPs++;
        }
        return numberMPs;
    }

    void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t &idx)
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    void KeyFrame::EraseMapPointMatch(const int &idx)
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
    {
        std::tuple<size_t, size_t> indexes = pMP->GetIndexInKeyFrame(this);
        size_t leftIndex = std::get<0>(indexes), rightIndex = std::get<1>(indexes);
        if(leftIndex != -1)
            mvpMapPoints[leftIndex] = static_cast<MapPoint*>(NULL);
        if(rightIndex != -1)
            mvpMapPoints[rightIndex] = static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
    {
        mvpMapPoints[idx] = pMP;
    }

    std::set<MapPoint*> KeyFrame::GetMapPoints()
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        std::set<MapPoint*> s;
        for(size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
        {
            if(!mvpMapPoints[i])
                continue;
            MapPoint* pMP = mvpMapPoints[i];
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }

    int KeyFrame::TrackedMapPoints(const int &minObs)
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);

        int nPoints = 0;
        const bool bCheckObs = minObs > 0;
        for(int i = 0; i < N; i++)
        {
            MapPoint* pMP = mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(bCheckObs)
                    {
                        if(mvpMapPoints[i]->Observations() >= minObs)
                            nPoints++;
                    }
                    else
                        nPoints++;
                }
            }
        }

        return nPoints;
    }

    std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }

    MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }

    void KeyFrame::UpdateConnections(bool upParent)
    {
        std::map<KeyFrame*, int> KFcounter;

        std::vector<MapPoint*> vpMP;

        {
            std::lock_guard<std::mutex> lockMPs(mMutexFeatures);
            vpMP = mvpMapPoints;
        }

        //For all map points in keyframe check in which other keyframes are they seen
        //Increase counter for those keyframes
        for(std::vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;

            if(!pMP)
                continue;

            if(pMP->isBad())
                continue;

            std::map<KeyFrame*, std::tuple<int, int>> observations = pMP->GetObservations();

            for(std::map<KeyFrame*, std::tuple<int, int>>::iterator mit = observations.begin(),
                                                                    mend = observations.end();
                mit != mend; mit++)
            {
                if(mit->first->mnId == mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
                    continue;
                KFcounter[mit->first]++;
            }
        }

        // This should not happen
        if(KFcounter.empty())
            return;

        //If the counter is greater than threshold add connection
        //In case no keyframe counter is over threshold add the one with maximum counter
        int nmax = 0;
        KeyFrame* pKFmax = NULL;
        int th = 15;

        std::vector<std::pair<int, KeyFrame*>> vPairs;
        vPairs.reserve(KFcounter.size());
        if(!upParent)
            std::cout << "UPDATE_CONN: current KF " << mnId << std::endl;
        for(std::map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
        {
            if(!upParent)
                std::cout << "  UPDATE_CONN: KF " << mit->first->mnId << " ; num matches: " << mit->second << std::endl;
            if(mit->second > nmax)
            {
                nmax = mit->second;
                pKFmax = mit->first;
            }
            if(mit->second >= th)
            {
                vPairs.push_back(std::make_pair(mit->second, mit->first));
                (mit->first)->AddConnection(this, mit->second);
            }
        }

        if(vPairs.empty())
        {
            vPairs.push_back(std::make_pair(nmax, pKFmax));
            pKFmax->AddConnection(this, nmax);
        }

        std::sort(vPairs.begin(), vPairs.end());
        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;
        for(size_t i = 0; i < vPairs.size(); i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        {
            std::lock_guard<std::mutex> lockCon(mMutexConnections);

            mConnectedKeyFrameWeights = KFcounter;
            mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
            mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

            if(mbFirstConnection && mnId != mpMap->GetInitKFid())
            {
                mpParent = mvpOrderedConnectedKeyFrames.front();
                mpParent->AddChild(this);
                mbFirstConnection = false;
            }
        }
    }

    void KeyFrame::AddChild(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        mspChildrens.insert(pKF);
    }

    void KeyFrame::EraseChild(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        mspChildrens.erase(pKF);
    }

    void KeyFrame::ChangeParent(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        if(pKF == this)
        {
            std::cout << "ERROR: Change parent KF, the parent and child are the same KF" << std::endl;
            throw std::invalid_argument("The parent and child can not be the same");
        }

        mpParent = pKF;
        pKF->AddChild(this);
    }

    std::set<KeyFrame*> KeyFrame::GetChilds()
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        return mspChildrens;
    }

    KeyFrame* KeyFrame::GetParent()
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        return mpParent;
    }

    bool KeyFrame::hasChild(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        return mspChildrens.count(pKF);
    }

    void KeyFrame::SetFirstConnection(bool bFirst)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        mbFirstConnection = bFirst;
    }

    void KeyFrame::AddLoopEdge(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        mbNotErase = true;
        mspLoopEdges.insert(pKF);
    }

    std::set<KeyFrame*> KeyFrame::GetLoopEdges()
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        return mspLoopEdges;
    }

    void KeyFrame::AddMergeEdge(KeyFrame* pKF)
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        mbNotErase = true;
        mspMergeEdges.insert(pKF);
    }

    std::set<KeyFrame*> KeyFrame::GetMergeEdges()
    {
        std::lock_guard<std::mutex> lockCon(mMutexConnections);
        return mspMergeEdges;
    }

    void KeyFrame::SetNotErase()
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        mbNotErase = true;
    }

    void KeyFrame::SetErase()
    {
        {
            std::lock_guard<std::mutex> lock(mMutexConnections);
            if(mspLoopEdges.empty())
            {
                mbNotErase = false;
            }
        }

        if(mbToBeErased)
        {
            SetBadFlag();
        }
    }

    void KeyFrame::SetBadFlag()
    {
        {
            std::lock_guard<std::mutex> lock(mMutexConnections);
            if(mnId == mpMap->GetInitKFid())
            {
                return;
            }
            else if(mbNotErase)
            {
                mbToBeErased = true;
                return;
            }
        }

        for(std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                               mend = mConnectedKeyFrameWeights.end();
            mit != mend; mit++)
        {
            mit->first->EraseConnection(this);
        }

        for(size_t i = 0; i < mvpMapPoints.size(); i++)
        {
            if(mvpMapPoints[i])
            {
                mvpMapPoints[i]->EraseObservation(this);
            }
        }

        {
            std::scoped_lock lock(mMutexConnections, mMutexFeatures);

            mConnectedKeyFrameWeights.clear();
            mvpOrderedConnectedKeyFrames.clear();

            // Update Spanning Tree
            std::set<KeyFrame*> sParentCandidates;
            if(mpParent)
                sParentCandidates.insert(mpParent);

            // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
            // Include that children as new parent candidate for the rest
            while(!mspChildrens.empty())
            {
                bool bContinue = false;

                int max = -1;
                KeyFrame* pC;
                KeyFrame* pP;

                for(std::set<KeyFrame*>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send;
                    sit++)
                {
                    KeyFrame* pKF = *sit;
                    if(pKF->isBad())
                        continue;

                    // Check if a parent candidate is connected to the keyframe
                    std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                    for(size_t i = 0, iend = vpConnected.size(); i < iend; i++)
                    {
                        for(std::set<KeyFrame*>::iterator spcit = sParentCandidates.begin(),
                                                          spcend = sParentCandidates.end();
                            spcit != spcend; spcit++)
                        {
                            if(vpConnected[i]->mnId == (*spcit)->mnId)
                            {
                                int w = pKF->GetWeight(vpConnected[i]);
                                if(w > max)
                                {
                                    pC = pKF;
                                    pP = vpConnected[i];
                                    max = w;
                                    bContinue = true;
                                }
                            }
                        }
                    }
                }

                if(bContinue)
                {
                    pC->ChangeParent(pP);
                    sParentCandidates.insert(pC);
                    mspChildrens.erase(pC);
                }
                else
                    break;
            }

            // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
            if(!mspChildrens.empty())
            {
                for(std::set<KeyFrame*>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
                {
                    (*sit)->ChangeParent(mpParent);
                }
            }

            if(mpParent)
            {
                mpParent->EraseChild(this);
                mTcp = mState.Tcw() * mpParent->GetPoseInverse();
            }
            mbBad = true;
        }

        mpMap->EraseKeyFrame(this);
        mpKeyFrameDB->erase(this);
    }

    bool KeyFrame::isBad()
    {
        std::lock_guard<std::mutex> lock(mMutexConnections);
        return mbBad;
    }

    void KeyFrame::EraseConnection(KeyFrame* pKF)
    {
        bool bUpdate = false;
        {
            std::lock_guard<std::mutex> lock(mMutexConnections);
            if(mConnectedKeyFrameWeights.count(pKF))
            {
                mConnectedKeyFrameWeights.erase(pKF);
                bUpdate = true;
            }
        }

        if(bUpdate)
            UpdateBestCovisibles();
    }

    std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r,
                                                    const bool bRight) const
    {
        std::vector<size_t> vIndices;
        vIndices.reserve(N);

        float factorX = r;
        float factorY = r;

        const int nMinCellX = std::max(0, (int)std::floor((x - mnMinX - factorX) * mfGridElementWidthInv));
        if(nMinCellX >= mnGridCols)
            return vIndices;

        const int nMaxCellX = std::min((int)mnGridCols - 1,
                                       (int)std::ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
        if(nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = std::max(0, (int)std::floor((y - mnMinY - factorY) * mfGridElementHeightInv));
        if(nMinCellY >= mnGridRows)
            return vIndices;

        const int nMaxCellY = std::min((int)mnGridRows - 1,
                                       (int)std::ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
        if(nMaxCellY < 0)
            return vIndices;

        for(int ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                const std::vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
                for(size_t j = 0, jend = vCell.size(); j < jend; j++)
                {
                    const cv::KeyPoint &kpUn = (NLeft == -1) ? mvKeysUn[vCell[j]]
                                               : (!bRight)   ? mvKeys[vCell[j]]
                                                             : mvKeysRight[vCell[j]];
                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if(std::fabs(distx) < r && std::fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    bool KeyFrame::IsInImage(const float &x, const float &y) const
    {
        return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
    }

    bool KeyFrame::UnprojectStereo(int i, Eigen::Vector3f &x3D)
    {
        const float z = mvDepth[i];
        if(z > 0)
        {
            const float u = mvKeys[i].pt.x;
            const float v = mvKeys[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            Eigen::Vector3f x3Dc(x, y, z);

            std::lock_guard<std::mutex> lock(mMutexPose);
            x3D = mState.Rwc() * x3Dc + mState.Ow();
            return true;
        }
        else
            return false;
    }

    float KeyFrame::ComputeSceneMedianDepth(const int q)
    {
        if(N == 0)
            return -1.0;

        std::vector<MapPoint*> vpMapPoints;
        Eigen::Matrix3f Rcw;
        Eigen::Vector3f tcw;
        {
            std::scoped_lock lock(mMutexFeatures, mMutexPose);
            vpMapPoints = mvpMapPoints;
            tcw = mState.tcw();
            Rcw = mState.Rcw();
        }

        std::vector<float> vDepths;
        vDepths.reserve(N);
        Eigen::Matrix<float, 1, 3> Rcw2 = Rcw.row(2);
        float zcw = tcw(2);
        for(int i = 0; i < N; i++)
        {
            if(mvpMapPoints[i])
            {
                MapPoint* pMP = mvpMapPoints[i];
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                float z = Rcw2.dot(x3Dw) + zcw;
                vDepths.push_back(z);
            }
        }

        std::sort(vDepths.begin(), vDepths.end());

        return vDepths[(vDepths.size() - 1) / q];
    }

    void KeyFrame::SetNewBias(const IMU::Bias &b)
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        mImuBias = b;
        if(mpImuPreintegrated)
            mpImuPreintegrated->SetNewBias(b);
    }

    Eigen::Vector3f KeyFrame::GetGyroBias()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
    }

    Eigen::Vector3f KeyFrame::GetAccBias()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
    }

    IMU::Bias KeyFrame::GetImuBias()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mImuBias;
    }

    Map* KeyFrame::GetMap()
    {
        std::lock_guard<std::mutex> lock(mMutexMap);
        return mpMap;
    }

    void KeyFrame::UpdateMap(Map* pMap)
    {
        std::lock_guard<std::mutex> lock(mMutexMap);
        mpMap = pMap;
    }

    void KeyFrame::PreSave(std::set<KeyFrame*> &spKF, std::set<MapPoint*> &spMP, std::set<GeometricCamera*> &spCam)
    {
        // Save the id of each MapPoint in this KF, there can be null pointer in the vector
        mvBackupMapPointsId.clear();
        mvBackupMapPointsId.reserve(N);
        for(int i = 0; i < N; ++i)
        {
            if(mvpMapPoints[i] && spMP.find(mvpMapPoints[i]) != spMP.end()) // Checks if the element is not null
                mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
            else // If the element is null his value is -1 because all the id are positives
                mvBackupMapPointsId.push_back(-1);
        }
        // Save the id of each connected KF with it weight
        mBackupConnectedKeyFrameIdWeights.clear();
        for(std::map<KeyFrame*, int>::const_iterator it = mConnectedKeyFrameWeights.begin(),
                                                     end = mConnectedKeyFrameWeights.end();
            it != end; ++it)
        {
            if(spKF.find(it->first) != spKF.end())
                mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
        }

        // Save the parent id
        mBackupParentId = -1;
        if(mpParent && spKF.find(mpParent) != spKF.end())
            mBackupParentId = mpParent->mnId;

        // Save the id of the childrens KF
        mvBackupChildrensId.clear();
        mvBackupChildrensId.reserve(mspChildrens.size());
        for(KeyFrame* pKFi : mspChildrens)
        {
            if(spKF.find(pKFi) != spKF.end())
                mvBackupChildrensId.push_back(pKFi->mnId);
        }

        // Save the id of the loop edge KF
        mvBackupLoopEdgesId.clear();
        mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
        for(KeyFrame* pKFi : mspLoopEdges)
        {
            if(spKF.find(pKFi) != spKF.end())
                mvBackupLoopEdgesId.push_back(pKFi->mnId);
        }

        // Save the id of the merge edge KF
        mvBackupMergeEdgesId.clear();
        mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
        for(KeyFrame* pKFi : mspMergeEdges)
        {
            if(spKF.find(pKFi) != spKF.end())
                mvBackupMergeEdgesId.push_back(pKFi->mnId);
        }

        //Camera data
        mnBackupIdCamera = -1;
        if(mpCamera && spCam.find(mpCamera) != spCam.end())
            mnBackupIdCamera = mpCamera->GetId();

        mnBackupIdCamera2 = -1;
        if(mpCamera2 && spCam.find(mpCamera2) != spCam.end())
            mnBackupIdCamera2 = mpCamera2->GetId();

        //Inertial data
        mBackupPrevKFId = -1;
        if(mPrevKF && spKF.find(mPrevKF) != spKF.end())
            mBackupPrevKFId = mPrevKF->mnId;

        mBackupNextKFId = -1;
        if(mNextKF && spKF.find(mNextKF) != spKF.end())
            mBackupNextKFId = mNextKF->mnId;

        if(mpImuPreintegrated)
            mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
    }

    void KeyFrame::PostLoad(std::map<long unsigned int, KeyFrame*> &mpKFid,
                            std::map<long unsigned int, MapPoint*> &mpMPid,
                            std::map<unsigned int, GeometricCamera*> &mpCamId)
    {
        // Rebuild the empty variables

        // Pose
        SetPose(mState.Tcw());

        mTrl = mTlr.inverse();

        // Reference reconstruction
        // Each MapPoint sight from this KeyFrame
        mvpMapPoints.clear();
        mvpMapPoints.resize(N);
        for(int i = 0; i < N; ++i)
        {
            if(mvBackupMapPointsId[i] != -1)
                mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
            else
                mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        // Conected KeyFrames with him weight
        mConnectedKeyFrameWeights.clear();
        for(std::map<long unsigned int, int>::const_iterator it = mBackupConnectedKeyFrameIdWeights.begin(),
                                                             end = mBackupConnectedKeyFrameIdWeights.end();
            it != end; ++it)
        {
            KeyFrame* pKFi = mpKFid[it->first];
            mConnectedKeyFrameWeights[pKFi] = it->second;
        }

        // Restore parent KeyFrame
        if(mBackupParentId >= 0)
            mpParent = mpKFid[mBackupParentId];

        // KeyFrame childrens
        mspChildrens.clear();
        for(std::vector<long unsigned int>::const_iterator it = mvBackupChildrensId.begin(),
                                                           end = mvBackupChildrensId.end();
            it != end; ++it)
        {
            mspChildrens.insert(mpKFid[*it]);
        }

        // Loop edge KeyFrame
        mspLoopEdges.clear();
        for(std::vector<long unsigned int>::const_iterator it = mvBackupLoopEdgesId.begin(),
                                                           end = mvBackupLoopEdgesId.end();
            it != end; ++it)
        {
            mspLoopEdges.insert(mpKFid[*it]);
        }

        // Merge edge KeyFrame
        mspMergeEdges.clear();
        for(std::vector<long unsigned int>::const_iterator it = mvBackupMergeEdgesId.begin(),
                                                           end = mvBackupMergeEdgesId.end();
            it != end; ++it)
        {
            mspMergeEdges.insert(mpKFid[*it]);
        }

        //Camera data
        if(mnBackupIdCamera >= 0)
        {
            mpCamera = mpCamId[mnBackupIdCamera];
        }
        else
        {
            std::cout << "ERROR: There is not a main camera in KF " << mnId << std::endl;
        }
        if(mnBackupIdCamera2 >= 0)
        {
            mpCamera2 = mpCamId[mnBackupIdCamera2];
        }

        //Inertial data
        if(mBackupPrevKFId != -1)
        {
            mPrevKF = mpKFid[mBackupPrevKFId];
        }
        if(mBackupNextKFId != -1)
        {
            mNextKF = mpKFid[mBackupNextKFId];
        }
        mpImuPreintegrated = &mBackupImuPreintegrated;

        // Remove all backup container
        mvBackupMapPointsId.clear();
        mBackupConnectedKeyFrameIdWeights.clear();
        mvBackupChildrensId.clear();
        mvBackupLoopEdgesId.clear();

        UpdateBestCovisibles();
    }

    Sophus::SE3f KeyFrame::GetRelativePoseTrl()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mTrl;
    }

    Sophus::SE3f KeyFrame::GetRelativePoseTlr()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return mTlr;
    }

    Sophus::SE3<float> KeyFrame::GetRightPose()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);

        return mTrl * mState.Tcw();
    }

    Sophus::SE3<float> KeyFrame::GetRightPoseInverse()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);

        return mState.Twc() * mTlr;
    }

    Eigen::Vector3f KeyFrame::GetRightCameraCenter()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);

        return (mState.Twc() * mTlr).translation();
    }

    Eigen::Matrix<float, 3, 3> KeyFrame::GetRightRotation()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);

        return (mTrl.so3() * mState.Tcw().so3()).matrix();
    }

    Eigen::Vector3f KeyFrame::GetRightTranslation()
    {
        std::lock_guard<std::mutex> lock(mMutexPose);
        return (mTrl * mState.Tcw()).translation();
    }

    void KeyFrame::SetORBVocabulary(ORBVocabulary* pORBVoc)
    {
        mpORBvocabulary = pORBVoc;
    }

    void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKFDB)
    {
        mpKeyFrameDB = pKFDB;
    }

} // namespace ORB_SLAM3
