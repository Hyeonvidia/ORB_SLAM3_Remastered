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

// See optimization/Shadow.hpp. Nothing of this file is in a build without
// ORBSLAM3R_OPT_SHADOW.
#ifdef ORBSLAM3R_OPT_SHADOW

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

#include <memory>

namespace ORB_SLAM3
{

    namespace shadow
    {

        // v1.0's Optimizer::LocalBundleAdjustment as it was, but that it says
        // which observations it erased and whether it got as far as solving.
        void LocalBundleAdjustmentV1(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, int &num_fixedKF, int &num_OptKF,
                                     int &num_MPs, int &num_edges,
                                     std::vector<std::pair<KeyFrame*, MapPoint*>> &vErased, bool &bSolved)
        {
            // Local KeyFrames: First Breath Search from Current Keyframe
            std::list<KeyFrame*> lLocalKeyFrames;

            lLocalKeyFrames.push_back(pKF);
            pKF->mnBALocalForKF = pKF->mnId;
            Map* pCurrentMap = pKF->GetMap();

            const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
            for(int i = 0, iend = vNeighKFs.size(); i < iend; i++)
            {
                KeyFrame* pKFi = vNeighKFs[i];
                pKFi->mnBALocalForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lLocalKeyFrames.push_back(pKFi);
            }

            // Local MapPoints seen in Local KeyFrames
            num_fixedKF = 0;
            std::list<MapPoint*> lLocalMapPoints;
            std::set<MapPoint*> sNumObsMP;
            for(std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend;
                lit++)
            {
                KeyFrame* pKFi = *lit;
                if(pKFi->mnId == pMap->GetInitKFid())
                {
                    num_fixedKF = 1;
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
                                lLocalMapPoints.push_back(pMP);
                                pMP->mnBALocalForKF = pKF->mnId;
                            }
                        }
                }
            }

            // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
            std::list<KeyFrame*> lFixedCameras;
            for(std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend;
                lit++)
            {
                MapPoint::ObservationMap observations = (*lit)->GetObservations();
                for(MapPoint::ObservationMap::iterator mit = observations.begin(), mend = observations.end();
                    mit != mend; mit++)
                {
                    KeyFrame* pKFi = mit->first;

                    if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                    {
                        pKFi->mnBAFixedForKF = pKF->mnId;
                        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                            lFixedCameras.push_back(pKFi);
                    }
                }
            }
            num_fixedKF = lFixedCameras.size() + num_fixedKF;

            if(num_fixedKF == 0)
            {
                Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted",
                                   Verbose::VERBOSITY_NORMAL);
                return;
            }

            // Setup optimizer
            g2o::SparseOptimizer optimizer;

            auto* solver =
                orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3, orbslam3r::g2o_ext::LinearSolver::kEigen>();
            if(pMap->IsInertial())
                solver->setUserLambdaInit(100.0);

            optimizer.setAlgorithm(solver);
            optimizer.setVerbose(false);

            if(pbStopFlag)
                optimizer.setForceStopFlag(pbStopFlag);

            unsigned long maxKFid = 0;

            // DEBUG LBA
            pCurrentMap->msOptKFs.clear();
            pCurrentMap->msFixedKFs.clear();

            // Set Local KeyFrame vertices
            for(std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend;
                lit++)
            {
                KeyFrame* pKFi = *lit;
                g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                Sophus::SE3<float> Tcw = pKFi->GetPose();
                vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
                vSE3->setId(pKFi->mnId);
                vSE3->setFixed(pKFi->mnId == pMap->GetInitKFid());
                optimizer.addVertex(vSE3);
                if(pKFi->mnId > maxKFid)
                    maxKFid = pKFi->mnId;
                // DEBUG LBA
                pCurrentMap->msOptKFs.insert(pKFi->mnId);
            }
            num_OptKF = lLocalKeyFrames.size();

            // Set Fixed KeyFrame vertices
            for(std::list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend;
                lit++)
            {
                KeyFrame* pKFi = *lit;
                g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                Sophus::SE3<float> Tcw = pKFi->GetPose();
                vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
                vSE3->setId(pKFi->mnId);
                vSE3->setFixed(true);
                optimizer.addVertex(vSE3);
                if(pKFi->mnId > maxKFid)
                    maxKFid = pKFi->mnId;
                // DEBUG LBA
                pCurrentMap->msFixedKFs.insert(pKFi->mnId);
            }

            // Set MapPoint vertices
            const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

            std::vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
            vpEdgesMono.reserve(nExpectedSize);

            std::vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
            vpEdgesBody.reserve(nExpectedSize);

            std::vector<KeyFrame*> vpEdgeKFMono;
            vpEdgeKFMono.reserve(nExpectedSize);

            std::vector<KeyFrame*> vpEdgeKFBody;
            vpEdgeKFBody.reserve(nExpectedSize);

            std::vector<MapPoint*> vpMapPointEdgeMono;
            vpMapPointEdgeMono.reserve(nExpectedSize);

            std::vector<MapPoint*> vpMapPointEdgeBody;
            vpMapPointEdgeBody.reserve(nExpectedSize);

            std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
            vpEdgesStereo.reserve(nExpectedSize);

            std::vector<KeyFrame*> vpEdgeKFStereo;
            vpEdgeKFStereo.reserve(nExpectedSize);

            std::vector<MapPoint*> vpMapPointEdgeStereo;
            vpMapPointEdgeStereo.reserve(nExpectedSize);

            const float thHuberMono = std::sqrt(5.991);
            const float thHuberStereo = std::sqrt(7.815);

            int nPoints = 0;

            int nEdges = 0;

            for(std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend;
                lit++)
            {
                MapPoint* pMP = *lit;
                g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
                vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
                int id = pMP->mnId + maxKFid + 1;
                vPoint->setId(id);
                vPoint->setMarginalized(true);
                optimizer.addVertex(vPoint);
                nPoints++;

                const MapPoint::ObservationMap observations = pMP->GetObservations();

                //Set edges
                for(MapPoint::ObservationMap::const_iterator mit = observations.begin(), mend = observations.end();
                    mit != mend; mit++)
                {
                    KeyFrame* pKFi = mit->first;

                    if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    {
                        const int leftIndex = std::get<0>(mit->second);

                        // Monocular observation
                        if(leftIndex != -1 && pKFi->mvuRight[std::get<0>(mit->second)] < 0)
                        {
                            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                            Eigen::Matrix<double, 2, 1> obs;
                            obs << kpUn.pt.x, kpUn.pt.y;

                            ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                            e->setMeasurement(obs);
                            const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);

                            e->pCamera = pKFi->mpCamera;

                            optimizer.addEdge(e);
                            vpEdgesMono.push_back(e);
                            vpEdgeKFMono.push_back(pKFi);
                            vpMapPointEdgeMono.push_back(pMP);

                            nEdges++;
                        }
                        else if(leftIndex != -1 && pKFi->mvuRight[std::get<0>(mit->second)] >= 0) // Stereo observation
                        {
                            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                            Eigen::Matrix<double, 3, 1> obs;
                            const float kp_ur = pKFi->mvuRight[std::get<0>(mit->second)];
                            obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                            g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                            e->setMeasurement(obs);
                            const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                            Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                            e->setInformation(Info);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberStereo);

                            e->fx = pKFi->fx;
                            e->fy = pKFi->fy;
                            e->cx = pKFi->cx;
                            e->cy = pKFi->cy;
                            e->bf = pKFi->mbf;

                            optimizer.addEdge(e);
                            vpEdgesStereo.push_back(e);
                            vpEdgeKFStereo.push_back(pKFi);
                            vpMapPointEdgeStereo.push_back(pMP);

                            nEdges++;
                        }

                        if(pKFi->mpCamera2)
                        {
                            int rightIndex = std::get<1>(mit->second);

                            if(rightIndex != -1)
                            {
                                rightIndex -= pKFi->NLeft;

                                Eigen::Matrix<double, 2, 1> obs;
                                cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                                obs << kp.pt.x, kp.pt.y;

                                ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                                e->setVertex(
                                    1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                                e->setMeasurement(obs);
                                const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(thHuberMono);

                                Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                                e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(),
                                                       Trl.translation().cast<double>());

                                e->pCamera = pKFi->mpCamera2;

                                optimizer.addEdge(e);
                                vpEdgesBody.push_back(e);
                                vpEdgeKFBody.push_back(pKFi);
                                vpMapPointEdgeBody.push_back(pMP);

                                nEdges++;
                            }
                        }
                    }
                }
            }
            num_edges = nEdges;

            if(pbStopFlag)
                if(*pbStopFlag)
                    return;

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
            if(!pMap->IsInertial() && (!vpEdgesStereo.empty() || !vpEdgesBody.empty()))
            {
                solver->setInitialDamping(1e-9);
                solver->setStallIterations(1);
            }

            optimizer.initializeOptimization();
            optimizer.optimize(10);

            std::vector<std::pair<KeyFrame*, MapPoint*>> vToErase;
            vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

            // Check inlier observations
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
                }
            }

            for(size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
                MapPoint* pMP = vpMapPointEdgeBody[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    KeyFrame* pKFi = vpEdgeKFBody[i];
                    vToErase.push_back(std::make_pair(pKFi, pMP));
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
                }
            }

            vErased = vToErase;
            bSolved = true;

            // Get Map Mutex
            std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

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

            // Recover optimized data
            //Keyframes
            for(std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend;
                lit++)
            {
                KeyFrame* pKFi = *lit;
                g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
                g2o::SE3Quat SE3quat = vSE3->estimate();
                Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
                pKFi->SetPose(Tiw);
            }

            //Points
            for(std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend;
                lit++)
            {
                MapPoint* pMP = *lit;
                g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(
                    optimizer.vertex(pMP->mnId + maxKFid + 1));
                pMP->SetWorldPos(vPoint->estimate().cast<float>());
                pMP->UpdateNormalAndDepth();
            }

            pMap->IncreaseChangeIndex();
        }

        void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, int &num_fixedKF, int &num_OptKF,
                                   int &num_MPs, int &num_edges)
        {
            // Neither is stopped part way: the flag would find the two at
            // different places.
            (void)pbStopFlag;

            LocalBaTask task;
            const bool bBuilt = task.Build(pKF, pMap);
            if(bBuilt)
            {
                const std::unique_ptr<optim::BundleAdjuster> pSolver = optim::MakeBundleAdjuster();
                task.Solve(*pSolver, nullptr);
                task.Classify();
            }
            task.ResetMarks();

            std::vector<std::pair<KeyFrame*, MapPoint*>> vErased;
            bool bSolved = false;
            LocalBundleAdjustmentV1(pKF, nullptr, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges, vErased, bSolved);

            bool same = bBuilt == bSolved && num_fixedKF == task.FixedKeyFrames();
            if(same && bBuilt)
                same = num_OptKF == task.LocalKeyFrames() && num_edges == task.Edges() && task.Matches(vErased);
            Count("LocalBundleAdjustment", same);
        }

    } // namespace shadow

} // namespace ORB_SLAM3

#endif // ORBSLAM3R_OPT_SHADOW
