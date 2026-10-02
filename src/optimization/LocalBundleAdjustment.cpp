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

#include "optimization/Optimizer.hpp"
#include "optimization/LocalBaTask.hpp"
#include "optimization/Shadow.hpp"
#include "tracking/Frame.hpp"

#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <orbslam3r/g2o_ext/compat.hpp>
#include <orbslam3r/g2o_ext/solver_factory.hpp>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "optimization/G2oTypes.hpp"
#include "common/Converter.hpp"

#include <mutex>

#include "optim_g2o/OptimizableTypes.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "optimization/KeyFrameAndPose.hpp"
#include "common/Verbose.hpp"

namespace ORB_SLAM3
{
    bool LocalBaTask::Build(KeyFrame* pKF, Map* pMap)
    {
        // Local KeyFrames: First Breath Search from Current Keyframe
        mvpLocalKF.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;
        Map* pCurrentMap = pKF->GetMap();

        const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i = 0, iend = vNeighKFs.size(); i < iend; i++)
        {
            KeyFrame* pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                mvpLocalKF.push_back(pKFi);
        }

        // Local MapPoints seen in Local KeyFrames
        mnFixedKF = 0;
        for(KeyFrame* pKFi : mvpLocalKF)
        {
            if(pKFi->mnId == pMap->GetInitKFid())
            {
                mnFixedKF = 1;
            }
            std::vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                    {
                        if(pMP->mnBALocalForKF != pKF->mnId)
                        {
                            mvpPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
            }
        }

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        for(MapPoint* pMP : mvpPoints)
        {
            MapPoint::ObservationMap observations = pMP->GetObservations();
            for(MapPoint::ObservationMap::iterator mit = observations.begin(), mend = observations.end(); mit != mend;
                mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                        mvpFixedKF.push_back(pKFi);
                }
            }
        }
        mnFixedKF = mvpFixedKF.size() + mnFixedKF;

        if(mnFixedKF == 0)
        {
            Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted",
                               Verbose::VERBOSITY_NORMAL);
            return false;
        }

        mbInertial = pMap->IsInertial();

        // DEBUG LBA
        pCurrentMap->msOptKFs.clear();
        pCurrentMap->msFixedKFs.clear();
        for(KeyFrame* pKFi : mvpLocalKF)
            pCurrentMap->msOptKFs.insert(pKFi->mnId);
        for(KeyFrame* pKFi : mvpFixedKF)
            pCurrentMap->msFixedKFs.insert(pKFi->mnId);

        // The poses, in the order of the keyframes' ids: that is the order
        // v1.0's graph had them in, the id of a vertex being the keyframe's.
        std::vector<std::pair<KeyFrame*, bool>> vPoses; // and whether the pose is held
        vPoses.reserve(mvpLocalKF.size() + mvpFixedKF.size());
        for(KeyFrame* pKFi : mvpLocalKF)
            vPoses.push_back(std::make_pair(pKFi, pKFi->mnId == pMap->GetInitKFid()));
        for(KeyFrame* pKFi : mvpFixedKF)
            vPoses.push_back(std::make_pair(pKFi, true));
        std::sort(vPoses.begin(), vPoses.end(),
                  [](const std::pair<KeyFrame*, bool> &a, const std::pair<KeyFrame*, bool> &b)
                  { return a.first->mnId < b.first->mnId; });

        std::unordered_map<KeyFrame*, int> mPoseOf;
        mPoseOf.reserve(vPoses.size());
        for(const std::pair<KeyFrame*, bool> &pose : vPoses)
        {
            KeyFrame* pKFi = pose.first;
            optim::Rig rig;
            rig.camera = pKFi->mpCamera;
            rig.camera2 = pKFi->mpCamera2;
            if(pKFi->mpCamera2)
            {
                Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                rig.Rrl = Trl.unit_quaternion().cast<double>();
                rig.trl = Trl.translation().cast<double>();
            }
            rig.fx = pKFi->fx;
            rig.fy = pKFi->fy;
            rig.cx = pKFi->cx;
            rig.cy = pKFi->cy;
            rig.bf = pKFi->mbf;

            Sophus::SE3<float> Tcw = pKFi->GetPose();
            mPoseOf[pKFi] = mProblem.addPose(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>(),
                                             pose.second, mProblem.addRig(rig));
        }
        mvnLocalPose.reserve(mvpLocalKF.size());
        for(KeyFrame* pKFi : mvpLocalKF)
            mvnLocalPose.push_back(mPoseOf[pKFi]);

        // The points likewise, in the order of their ids.
        std::vector<int> vOrder(mvpPoints.size());
        for(std::size_t i = 0; i < vOrder.size(); i++)
            vOrder[i] = i;
        std::sort(vOrder.begin(), vOrder.end(),
                  [this](int a, int b) { return mvpPoints[a]->mnId < mvpPoints[b]->mnId; });
        mvnPoint.resize(mvpPoints.size());
        for(int i : vOrder)
            mvnPoint[i] = mProblem.addPoint(mvpPoints[i]->GetWorldPos().cast<double>());

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        // The observations, in the order v1.0 added their edges: point by
        // point as the window was walked, and for each point keyframe by
        // keyframe.
        for(std::size_t i = 0; i < mvpPoints.size(); i++)
        {
            MapPoint* pMP = mvpPoints[i];
            const MapPoint::ObservationMap observations = pMP->GetObservations();

            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(), mend = observations.end();
                mit != mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const std::unordered_map<KeyFrame*, int>::const_iterator itPose = mPoseOf.find(pKFi);
                    if(itPose == mPoseOf.end())
                        continue;
                    const int leftIndex = std::get<0>(mit->second);

                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[std::get<0>(mit->second)] < 0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        mProblem.addObservation(optim::kMono, itPose->second, mvnPoint[i],
                                                Eigen::Vector3d(kpUn.pt.x, kpUn.pt.y, 0), invSigma2, thHuberMono);
                        mvpObsKF.push_back(pKFi);
                        mvpObsMP.push_back(pMP);
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[std::get<0>(mit->second)] >= 0) // Stereo observation
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        const float kp_ur = pKFi->mvuRight[std::get<0>(mit->second)];
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        mProblem.addObservation(optim::kStereo, itPose->second, mvnPoint[i],
                                                Eigen::Vector3d(kpUn.pt.x, kpUn.pt.y, kp_ur), invSigma2, thHuberStereo);
                        mvpObsKF.push_back(pKFi);
                        mvpObsMP.push_back(pMP);
                        mbDepth = true;
                    }

                    if(pKFi->mpCamera2)
                    {
                        int rightIndex = std::get<1>(mit->second);

                        if(rightIndex != -1)
                        {
                            rightIndex -= pKFi->NLeft;

                            cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                            const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                            mProblem.addObservation(optim::kRight, itPose->second, mvnPoint[i],
                                                    Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), invSigma2, thHuberMono);
                            mvpObsKF.push_back(pKFi);
                            mvpObsMP.push_back(pMP);
                            mbDepth = true;
                        }
                    }
                }
            }
        }

        return true;
    }

    void LocalBaTask::Solve(optim::BundleAdjuster &solver, bool* pbStopFlag)
    {
        optim::SolveOptions options;
        options.nIterations = 10;
        options.pbStop = pbStopFlag;

        if(mbInertial)
        {
            options.damping = optim::SolveOptions::kValue;
            options.dampingValue = 100.0;
        }
        // Where depth is measured -- stereo, RGB-D -- the damping starts small
        // and the adjustment stops at the first iteration that gains less than
        // a thousandth. With g2o's initial damping it ran its ten iterations
        // and, on KITTI stereo, ended where it is after two with this one;
        // with this one it has converged after four, and the iterations after
        // the first small gain gained less still.
        //
        // Not in monocular, where it was measured to cost accuracy (KITTI 07:
        // an ATE of 3.0 to 4.0 m where it had been 2.0 to 3.0): nothing but
        // the fixed keyframes holds the scale of a monocular window, and the
        // damping is what keeps an iteration from moving far along it.
        else if(mbDepth)
        {
            options.damping = optim::SolveOptions::kFractionOfDiagonal;
            options.dampingValue = 1e-9;
            options.nStallIterations = 1;
        }

        solver.Prepare(mProblem);
        solver.Solve(mProblem, options);
    }

    const std::vector<std::pair<KeyFrame*, MapPoint*>> &LocalBaTask::Classify()
    {
        mvToErase.clear();
        mvToErase.reserve(mProblem.observations());

        // Check inlier observations: the first camera's, the second's, then
        // the stereo ones, which is the order v1.0 erased them in.
        for(const optim::ObservationKind kind : {optim::kMono, optim::kRight, optim::kStereo})
        {
            const double chi2Max = kind == optim::kStereo ? 7.815 : 5.991;
            for(std::size_t i = 0, iend = mProblem.observations(); i < iend; i++)
            {
                if(mProblem.kind[i] != kind)
                    continue;
                MapPoint* pMP = mvpObsMP[i];

                if(pMP->isBad())
                    continue;

                if(mProblem.chi2[i] > chi2Max || !mProblem.depthPositive[i])
                    mvToErase.push_back(std::make_pair(mvpObsKF[i], pMP));
            }
        }
        return mvToErase;
    }

    void LocalBaTask::Apply(Map* pMap)
    {
        Classify();

        // Get Map Mutex
        std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

        if(!mvToErase.empty())
        {
            for(size_t i = 0; i < mvToErase.size(); i++)
            {
                KeyFrame* pKFi = mvToErase[i].first;
                MapPoint* pMPi = mvToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        // Recover optimized data
        //Keyframes
        for(std::size_t i = 0; i < mvpLocalKF.size(); i++)
        {
            const int n = mvnLocalPose[i];
            Sophus::SE3f Tiw(mProblem.Rcw[n].cast<float>(), mProblem.tcw[n].cast<float>());
            mvpLocalKF[i]->SetPose(Tiw);
        }

        //Points
        for(std::size_t i = 0; i < mvpPoints.size(); i++)
        {
            MapPoint* pMP = mvpPoints[i];
            pMP->SetWorldPos(mProblem.Xw[mvnPoint[i]].cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    }

    bool LocalBaTask::Matches(const std::vector<std::pair<KeyFrame*, MapPoint*>> &erased) const
    {
        if(erased != mvToErase)
            return false;
        for(std::size_t i = 0; i < mvpLocalKF.size(); i++)
        {
            const int n = mvnLocalPose[i];
            const Sophus::SE3f Tiw(mProblem.Rcw[n].cast<float>(), mProblem.tcw[n].cast<float>());
            const Sophus::SE3f Tmap = mvpLocalKF[i]->GetPose();
            if(std::memcmp(Tiw.data(), Tmap.data(), 7 * sizeof(float)) != 0)
                return false;
        }
        for(std::size_t i = 0; i < mvpPoints.size(); i++)
        {
            const Eigen::Vector3f X = mProblem.Xw[mvnPoint[i]].cast<float>();
            const Eigen::Vector3f Xmap = mvpPoints[i]->GetWorldPos();
            if(std::memcmp(X.data(), Xmap.data(), 3 * sizeof(float)) != 0)
                return false;
        }
        return true;
    }

    void LocalBaTask::ResetMarks()
    {
        // No keyframe has this id, so v1.0's body takes none of them as seen.
        const long unsigned int none = ~0ul;
        for(KeyFrame* pKFi : mvpLocalKF)
            pKFi->mnBALocalForKF = none;
        for(KeyFrame* pKFi : mvpFixedKF)
            pKFi->mnBAFixedForKF = none;
        for(MapPoint* pMP : mvpPoints)
            pMP->mnBALocalForKF = none;
    }

    void Optimizer::LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, int &num_fixedKF, int &num_OptKF,
                                          int &num_MPs, int &num_edges)
    {
#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::LocalBundleAdjustment(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges);
#else
        LocalBaTask task;
        const bool bBuilt = task.Build(pKF, pMap);
        num_fixedKF = task.FixedKeyFrames();
        if(!bBuilt)
            return;
        num_OptKF = task.LocalKeyFrames();
        num_edges = task.Edges();

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        const std::unique_ptr<optim::BundleAdjuster> pSolver = optim::MakeBundleAdjuster();
        task.Solve(*pSolver, pbStopFlag);
        task.Apply(pMap);
#endif
    }

    void Optimizer::LocalBundleAdjustment(KeyFrame* pMainKF, std::vector<KeyFrame*> vpAdjustKF,
                                          std::vector<KeyFrame*> vpFixedKF, bool* pbStopFlag)
    {
        bool bShowImages = false;

        std::vector<MapPoint*> vpMPs;

        g2o::SparseOptimizer optimizer;

        auto* solver =
            orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3, orbslam3r::g2o_ext::LinearSolver::kEigen>();
        optimizer.setAlgorithm(solver);

        optimizer.setVerbose(false);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;
        std::set<KeyFrame*> spKeyFrameBA;

        Map* pCurrentMap = pMainKF->GetMap();

        // Set fixed KeyFrame vertices
        int numInsertedPoints = 0;
        for(KeyFrame* pKFi : vpFixedKF)
        {
            if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            {
                Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
                continue;
            }

            pKFi->mnBALocalForMerge = pMainKF->mnId;

            g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;

            std::set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
            for(MapPoint* pMPi : spViewMPs)
            {
                if(pMPi)
                    if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

                        if(pMPi->mnBALocalForMerge != pMainKF->mnId)
                        {
                            vpMPs.push_back(pMPi);
                            pMPi->mnBALocalForMerge = pMainKF->mnId;
                            numInsertedPoints++;
                        }
            }

            spKeyFrameBA.insert(pKFi);
        }

        // Set non fixed Keyframe vertices
        std::set<KeyFrame*> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
        numInsertedPoints = 0;
        for(KeyFrame* pKFi : vpAdjustKF)
        {
            if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;

            pKFi->mnBALocalForMerge = pMainKF->mnId;

            g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;

            std::set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
            for(MapPoint* pMPi : spViewMPs)
            {
                if(pMPi)
                {
                    if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
                    {
                        if(pMPi->mnBALocalForMerge != pMainKF->mnId)
                        {
                            vpMPs.push_back(pMPi);
                            pMPi->mnBALocalForMerge = pMainKF->mnId;
                            numInsertedPoints++;
                        }
                    }
                }
            }

            spKeyFrameBA.insert(pKFi);
        }

        const int nExpectedSize = (vpAdjustKF.size() + vpFixedKF.size()) * vpMPs.size();

        std::vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        std::vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        std::vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        std::vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        std::vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuber2D = std::sqrt(5.99);
        const float thHuber3D = std::sqrt(7.815);

        // Set MapPoint vertices
        std::map<KeyFrame*, int> mpObsKFs;
        std::map<KeyFrame*, int> mpObsFinalKFs;
        std::map<MapPoint*, int> mpObsMPs;
        for(unsigned int i = 0; i < vpMPs.size(); ++i)
        {
            MapPoint* pMPi = vpMPs[i];
            if(pMPi->isBad())
                continue;

            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(pMPi->GetWorldPos().cast<double>());
            const int id = pMPi->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const MapPoint::ObservationMap observations = pMPi->GetObservations();
            int nEdges = 0;
            //SET EDGES
            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId ||
                   !pKF->GetMapPoint(std::get<0>(mit->second)))
                    continue;

                nEdges++;

                const cv::KeyPoint &kpUn = pKF->mvKeysUn[std::get<0>(mit->second)];

                if(pKF->mvuRight[std::get<0>(mit->second)] < 0) //Monocular
                {
                    mpObsMPs[pMPi]++;
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    e->pCamera = pKF->mpCamera;

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKF);
                    vpMapPointEdgeMono.push_back(pMPi);

                    mpObsKFs[pKF]++;
                }
                else // RGBD or Stereo
                {
                    mpObsMPs[pMPi] += 2;
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[std::get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKF);
                    vpMapPointEdgeStereo.push_back(pMPi);

                    mpObsKFs[pKF]++;
                }
            }
        }

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;

        if(pbStopFlag)
            if(*pbStopFlag)
                bDoMore = false;

        std::map<unsigned long int, int> mWrongObsKF;
        if(bDoMore)
        {
            // Check inlier observations
            int badMonoMP = 0, badStereoMP = 0;
            for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
                MapPoint* pMP = vpMapPointEdgeMono[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                    badMonoMP++;
                }
                e->setRobustKernel(0);
            }

            for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
                MapPoint* pMP = vpMapPointEdgeStereo[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2() > 7.815 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                    badStereoMP++;
                }

                e->setRobustKernel(0);
            }
            Verbose::PrintMess("[BA]: First optimization(Huber), there are " + std::to_string(badMonoMP) +
                                   " monocular and " + std::to_string(badStereoMP) + " stereo bad edges",
                               Verbose::VERBOSITY_DEBUG);

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        std::vector<std::pair<KeyFrame*, MapPoint*>> vToErase;
        vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());
        std::set<MapPoint*> spErasedMPs;
        std::set<KeyFrame*> spErasedKFs;

        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2() > 5.991 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFMono[i];
                vToErase.push_back(std::make_pair(pKFi, pMP));
                mWrongObsKF[pKFi->mnId]++;
                badMonoMP++;

                spErasedMPs.insert(pMP);
                spErasedKFs.insert(pKFi);
            }
        }

        for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2() > 7.815 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(std::make_pair(pKFi, pMP));
                mWrongObsKF[pKFi->mnId]++;
                badStereoMP++;

                spErasedMPs.insert(pMP);
                spErasedKFs.insert(pKFi);
            }
        }

        Verbose::PrintMess("[BA]: Second optimization, there are " + std::to_string(badMonoMP) + " monocular and " +
                               std::to_string(badStereoMP) + " sterero bad edges",
                           Verbose::VERBOSITY_DEBUG);

        // Get Map Mutex
        std::lock_guard<std::mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

        if(!vToErase.empty())
        {
            for(size_t i = 0; i < vToErase.size(); i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        for(unsigned int i = 0; i < vpMPs.size(); ++i)
        {
            MapPoint* pMPi = vpMPs[i];
            if(pMPi->isBad())
                continue;

            const MapPoint::ObservationMap observations = pMPi->GetObservations();
            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForKF != pMainKF->mnId ||
                   !pKF->GetMapPoint(std::get<0>(mit->second)))
                    continue;

                if(pKF->mvuRight[std::get<0>(mit->second)] < 0) //Monocular
                {
                    mpObsFinalKFs[pKF]++;
                }
                else // RGBD or Stereo
                {
                    mpObsFinalKFs[pKF]++;
                }
            }
        }

        // Recover optimized data
        // Keyframes
        for(KeyFrame* pKFi : vpAdjustKF)
        {
            if(pKFi->isBad())
                continue;

            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());

            int numMonoBadPoints = 0, numMonoOptPoints = 0;
            int numStereoBadPoints = 0, numStereoOptPoints = 0;
            std::vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;
            std::vector<MapPoint*> vpMonoMPsBad, vpStereoMPsBad;

            for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
                MapPoint* pMP = vpMapPointEdgeMono[i];
                KeyFrame* pKFedge = vpEdgeKFMono[i];

                if(pKFi != pKFedge)
                {
                    continue;
                }

                if(pMP->isBad())
                    continue;

                if(e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    numMonoBadPoints++;
                    vpMonoMPsBad.push_back(pMP);
                }
                else
                {
                    numMonoOptPoints++;
                    vpMonoMPsOpt.push_back(pMP);
                }
            }

            for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
                MapPoint* pMP = vpMapPointEdgeStereo[i];
                KeyFrame* pKFedge = vpEdgeKFMono[i];

                if(pKFi != pKFedge)
                {
                    continue;
                }

                if(pMP->isBad())
                    continue;

                if(e->chi2() > 7.815 || !e->isDepthPositive())
                {
                    numStereoBadPoints++;
                    vpStereoMPsBad.push_back(pMP);
                }
                else
                {
                    numStereoOptPoints++;
                    vpStereoMPsOpt.push_back(pMP);
                }
            }

            pKFi->SetPose(Tiw);
        }

        //Points
        for(MapPoint* pMPi : vpMPs)
        {
            if(pMPi->isBad())
                continue;

            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(
                optimizer.vertex(pMPi->mnId + maxKFid + 1));
            pMPi->SetWorldPos(vPoint->estimate().cast<float>());
            pMPi->UpdateNormalAndDepth();
        }
    }

} // namespace ORB_SLAM3
