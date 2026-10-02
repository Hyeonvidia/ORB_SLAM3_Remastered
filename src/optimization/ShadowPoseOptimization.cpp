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
#include "optimization/PoseTask.hpp"
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

#include <atomic>
#include <cstdio>
#include <cstring>

namespace ORB_SLAM3
{

    namespace shadow
    {

        // v1.0's Optimizer::PoseOptimization as it was, but for the lock, which
        // the caller holds.
        int PoseOptimizationV1(Frame* pFrame)
        {
            g2o::SparseOptimizer optimizer;

            auto* solver =
                orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3, orbslam3r::g2o_ext::LinearSolver::kDense>();
            optimizer.setAlgorithm(solver);

            int nInitialCorrespondences = 0;

            // Set Frame vertex
            g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pFrame->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(0);
            vSE3->setFixed(false);
            optimizer.addVertex(vSE3);

            // Set MapPoint vertices
            const int N = pFrame->N;

            std::vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
            std::vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody*> vpEdgesMono_FHR;
            std::vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
            vpEdgesMono.reserve(N);
            vpEdgesMono_FHR.reserve(N);
            vnIndexEdgeMono.reserve(N);
            vnIndexEdgeRight.reserve(N);

            std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
            std::vector<size_t> vnIndexEdgeStereo;
            vpEdgesStereo.reserve(N);
            vnIndexEdgeStereo.reserve(N);

            const float deltaMono = std::sqrt(5.991);
            const float deltaStereo = std::sqrt(7.815);

            {
                // The caller holds MapPoint::mGlobalMutex.

                for(int i = 0; i < N; i++)
                {
                    MapPoint* pMP = pFrame->mvpMapPoints[i];
                    if(pMP)
                    {
                        //Conventional SLAM
                        if(!pFrame->mpCamera2)
                        {
                            // Monocular observation
                            if(pFrame->mvuRight[i] < 0)
                            {
                                nInitialCorrespondences++;
                                pFrame->mvbOutlier[i] = false;

                                Eigen::Matrix<double, 2, 1> obs;
                                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                                obs << kpUn.pt.x, kpUn.pt.y;

                                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                                e->setMeasurement(obs);
                                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(deltaMono);

                                e->pCamera = pFrame->mpCamera;
                                e->Xw = pMP->GetWorldPos().cast<double>();

                                optimizer.addEdge(e);

                                vpEdgesMono.push_back(e);
                                vnIndexEdgeMono.push_back(i);
                            }
                            else // Stereo observation
                            {
                                nInitialCorrespondences++;
                                pFrame->mvbOutlier[i] = false;

                                Eigen::Matrix<double, 3, 1> obs;
                                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                                const float &kp_ur = pFrame->mvuRight[i];
                                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                                e->setMeasurement(obs);
                                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                                e->setInformation(Info);

                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(deltaStereo);

                                e->fx = pFrame->fx;
                                e->fy = pFrame->fy;
                                e->cx = pFrame->cx;
                                e->cy = pFrame->cy;
                                e->bf = pFrame->mbf;
                                e->Xw = pMP->GetWorldPos().cast<double>();

                                optimizer.addEdge(e);

                                vpEdgesStereo.push_back(e);
                                vnIndexEdgeStereo.push_back(i);
                            }
                        }
                        //SLAM with respect a rigid body
                        else
                        {
                            nInitialCorrespondences++;

                            cv::KeyPoint kpUn;

                            if(i < pFrame->Nleft)
                            { //Left camera observation
                                kpUn = pFrame->mvKeys[i];

                                pFrame->mvbOutlier[i] = false;

                                Eigen::Matrix<double, 2, 1> obs;
                                obs << kpUn.pt.x, kpUn.pt.y;

                                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                                e->setMeasurement(obs);
                                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(deltaMono);

                                e->pCamera = pFrame->mpCamera;
                                e->Xw = pMP->GetWorldPos().cast<double>();

                                optimizer.addEdge(e);

                                vpEdgesMono.push_back(e);
                                vnIndexEdgeMono.push_back(i);
                            }
                            else
                            {
                                kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                                Eigen::Matrix<double, 2, 1> obs;
                                obs << kpUn.pt.x, kpUn.pt.y;

                                pFrame->mvbOutlier[i] = false;

                                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e =
                                    new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                                e->setMeasurement(obs);
                                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(deltaMono);

                                e->pCamera = pFrame->mpCamera2;
                                e->Xw = pMP->GetWorldPos().cast<double>();

                                e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(),
                                                       pFrame->GetRelativePoseTrl().translation().cast<double>());

                                optimizer.addEdge(e);

                                vpEdgesMono_FHR.push_back(e);
                                vnIndexEdgeRight.push_back(i);
                            }
                        }
                    }
                }
            }

            if(nInitialCorrespondences < 3)
                return 0;

            // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
            // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
            const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
            const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
            const int its[4] = {10, 10, 10, 10};

            int nBad = 0;
            for(size_t it = 0; it < 4; it++)
            {
                Tcw = pFrame->GetPose();
                vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));

                optimizer.initializeOptimization(0);
                optimizer.optimize(its[it]);

                nBad = 0;
                for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

                    const size_t idx = vnIndexEdgeMono[i];

                    if(pFrame->mvbOutlier[idx])
                    {
                        e->computeError();
                    }

                    const float chi2 = e->chi2();

                    if(chi2 > chi2Mono[it])
                    {
                        pFrame->mvbOutlier[idx] = true;
                        e->setLevel(1);
                        nBad++;
                    }
                    else
                    {
                        pFrame->mvbOutlier[idx] = false;
                        e->setLevel(0);
                    }

                    if(it == 2)
                        e->setRobustKernel(0);
                }

                for(size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

                    const size_t idx = vnIndexEdgeRight[i];

                    if(pFrame->mvbOutlier[idx])
                    {
                        e->computeError();
                    }

                    const float chi2 = e->chi2();

                    if(chi2 > chi2Mono[it])
                    {
                        pFrame->mvbOutlier[idx] = true;
                        e->setLevel(1);
                        nBad++;
                    }
                    else
                    {
                        pFrame->mvbOutlier[idx] = false;
                        e->setLevel(0);
                    }

                    if(it == 2)
                        e->setRobustKernel(0);
                }

                for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
                {
                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

                    const size_t idx = vnIndexEdgeStereo[i];

                    if(pFrame->mvbOutlier[idx])
                    {
                        e->computeError();
                    }

                    const float chi2 = e->chi2();

                    if(chi2 > chi2Stereo[it])
                    {
                        pFrame->mvbOutlier[idx] = true;
                        e->setLevel(1);
                        nBad++;
                    }
                    else
                    {
                        e->setLevel(0);
                        pFrame->mvbOutlier[idx] = false;
                    }

                    if(it == 2)
                        e->setRobustKernel(0);
                }

                if(optimizer.edges().size() < 10)
                    break;
            }

            // Recover optimized pose and return number of inliers
            g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
            g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
            Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(), SE3quat_recov.translation().cast<float>());
            pFrame->SetPose(pose);

            return nInitialCorrespondences - nBad;
        }

        int PoseOptimization(Frame* pFrame)
        {
            // Held across both, so that no point moves between them.
            std::lock_guard<std::mutex> lock(MapPoint::mGlobalMutex);

            const Sophus::SE3f poseBefore = pFrame->GetPose();
            const std::vector<bool> outliersBefore = pFrame->mvbOutlier;

            PoseTask task;
            task.Build(pFrame);
            const std::unique_ptr<optim::PoseSolver> pSolver = optim::MakePoseSolver();
            task.Solve(*pSolver);
            const int nNow = task.Apply(pFrame);
            const Sophus::SE3f poseNow = pFrame->GetPose();
            const std::vector<bool> outliersNow = pFrame->mvbOutlier;

            pFrame->SetPose(poseBefore);
            pFrame->mvbOutlier = outliersBefore;
            const int nV1 = PoseOptimizationV1(pFrame);
            const Sophus::SE3f poseV1 = pFrame->GetPose();

            const bool same = nNow == nV1 && outliersNow == pFrame->mvbOutlier &&
                              std::memcmp(poseNow.data(), poseV1.data(), 7 * sizeof(float)) == 0;
            Count("PoseOptimization", same);
            return nV1;
        }

    } // namespace shadow

} // namespace ORB_SLAM3

#endif // ORBSLAM3R_OPT_SHADOW
