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
#include "optimization/GlobalBaTask.hpp"
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
    void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF,
                                           const bool bRobust)
    {
        std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        std::vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
        BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    void GlobalBaTask::Build(const std::vector<KeyFrame*> &vpKFs, const std::vector<MapPoint*> &vpMP,
                             const bool bRobust)
    {
        mpMap = vpKFs[0]->GetMap();
        mvpKF = vpKFs;
        mvpMP = vpMP;

        long unsigned int maxKFid = 0;

        // The poses in the order of the keyframes' ids, as v1.0's graph had
        // them.
        for(KeyFrame* pKF : vpKFs)
        {
            if(pKF->isBad())
                continue;
            mvpPoseKF.push_back(pKF);
            if(pKF->mnId > maxKFid)
                maxKFid = pKF->mnId;
        }
        std::stable_sort(mvpPoseKF.begin(), mvpPoseKF.end(),
                         [](const KeyFrame* a, const KeyFrame* b) { return a->mnId < b->mnId; });

        std::unordered_map<KeyFrame*, int> mPoseOf;
        mPoseOf.reserve(mvpPoseKF.size());
        std::vector<KeyFrame*> vpUnique;
        vpUnique.reserve(mvpPoseKF.size());
        for(KeyFrame* pKF : mvpPoseKF)
        {
            if(mPoseOf.count(pKF))
                continue;
            optim::Rig rig;
            rig.camera = pKF->mpCamera;
            rig.camera2 = pKF->mpCamera2;
            if(pKF->mpCamera2)
            {
                Sophus::SE3f Trl = pKF->GetRelativePoseTrl();
                rig.Rrl = Trl.unit_quaternion().cast<double>();
                rig.trl = Trl.translation().cast<double>();
            }
            rig.fx = pKF->fx;
            rig.fy = pKF->fy;
            rig.cx = pKF->cx;
            rig.cy = pKF->cy;
            rig.bf = pKF->mbf;

            Sophus::SE3<float> Tcw = pKF->GetPose();
            mPoseOf[pKF] = mProblem.addPose(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>(),
                                            pKF->mnId == mpMap->GetInitKFid(), mProblem.addRig(rig));
            vpUnique.push_back(pKF);
        }
        mvpPoseKF.swap(vpUnique);
        mvnPose.assign(vpKFs.size(), -1);
        for(std::size_t i = 0; i < vpKFs.size(); i++)
        {
            const std::unordered_map<KeyFrame*, int>::const_iterator it = mPoseOf.find(vpKFs[i]);
            if(it != mPoseOf.end())
                mvnPose[i] = it->second;
        }

        const float thHuber2D = std::sqrt(5.99);
        const float thHuber3D = std::sqrt(7.815);

        // The observations, point by point as the points were given. A point
        // enters the problem if a keyframe of it observes it; which place it
        // gets there is known only when all have been walked, so its
        // observations wait with its place in the list.
        struct Observation
        {
            optim::ObservationKind kind;
            int pose;
            int listed;
            Eigen::Vector3d uv;
            double invSigma2;
            double huber;
            bool robust;
        };
        std::vector<Observation> vObservations;
        std::vector<Eigen::Vector3d> vXw(vpMP.size());
        std::vector<unsigned char> vbIncluded(vpMP.size(), 0);

        for(size_t i = 0; i < vpMP.size(); i++)
        {
            MapPoint* pMP = vpMP[i];
            if(pMP->isBad())
                continue;
            vXw[i] = pMP->GetWorldPos().cast<double>();

            const MapPoint::ObservationMap observations = pMP->GetObservations();

            int nEdges = 0;
            //SET EDGES
            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF->isBad() || pKF->mnId > maxKFid)
                    continue;
                const std::unordered_map<KeyFrame*, int>::const_iterator itPose = mPoseOf.find(pKF);
                if(itPose == mPoseOf.end())
                    continue;
                nEdges++;

                const int leftIndex = std::get<0>(mit->second);

                if(leftIndex != -1 && pKF->mvuRight[std::get<0>(mit->second)] < 0)
                {
                    const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    vObservations.push_back({optim::kMono, itPose->second, static_cast<int>(i),
                                             Eigen::Vector3d(kpUn.pt.x, kpUn.pt.y, 0), invSigma2, thHuber2D, bRobust});
                }
                else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
                {
                    const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];
                    const float kp_ur = pKF->mvuRight[std::get<0>(mit->second)];
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    vObservations.push_back({optim::kStereo, itPose->second, static_cast<int>(i),
                                             Eigen::Vector3d(kpUn.pt.x, kpUn.pt.y, kp_ur), invSigma2, thHuber3D,
                                             bRobust});
                }

                if(pKF->mpCamera2)
                {
                    int rightIndex = std::get<1>(mit->second);

                    if(rightIndex != -1 && rightIndex < pKF->mvKeysRight.size())
                    {
                        rightIndex -= pKF->NLeft;

                        cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                        const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                        vObservations.push_back({optim::kRight, itPose->second, static_cast<int>(i),
                                                 Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), invSigma2, thHuber2D, true});
                    }
                }
            }

            vbIncluded[i] = nEdges != 0;
        }

        // The points in the order of their ids.
        std::vector<int> vOrder;
        vOrder.reserve(vpMP.size());
        for(std::size_t i = 0; i < vpMP.size(); i++)
            if(vbIncluded[i])
                vOrder.push_back(i);
        std::sort(vOrder.begin(), vOrder.end(), [&vpMP](int a, int b) { return vpMP[a]->mnId < vpMP[b]->mnId; });
        mvnPoint.assign(vpMP.size(), -1);
        mvpPointMP.reserve(vOrder.size());
        for(int i : vOrder)
        {
            mvnPoint[i] = mProblem.addPoint(vXw[i]);
            mvpPointMP.push_back(vpMP[i]);
        }

        for(const Observation &o : vObservations)
            mProblem.addObservation(o.kind, o.pose, mvnPoint[o.listed], o.uv, o.invSigma2, o.huber, o.robust);
    }

    void GlobalBaTask::Solve(optim::BundleAdjuster &solver, int nIterations, bool* pbStopFlag)
    {
        optim::SolveOptions options;
        options.nIterations = nIterations;
        options.pbStop = pbStopFlag;

        solver.Prepare(mProblem);
        solver.Solve(mProblem, options);
    }

    void GlobalBaTask::Apply(const unsigned long nLoopKF) const
    {
        // Recover optimized data
        //Keyframes
        for(size_t i = 0; i < mvpKF.size(); i++)
        {
            KeyFrame* pKF = mvpKF[i];
            if(pKF->isBad())
                continue;
            const int n = mvnPose[i];
            if(n < 0)
                continue;

            if(nLoopKF == mpMap->GetOriginKF()->mnId)
            {
                pKF->SetPose(Sophus::SE3f(mProblem.Rcw[n].cast<float>(), mProblem.tcw[n].cast<float>()));
            }
            else
            {
                pKF->mTcwGBA = Sophus::SE3d(mProblem.Rcw[n], mProblem.tcw[n]).cast<float>();
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Points
        for(size_t i = 0; i < mvpMP.size(); i++)
        {
            if(mvnPoint[i] < 0)
                continue;

            MapPoint* pMP = mvpMP[i];

            if(pMP->isBad())
                continue;

            if(nLoopKF == mpMap->GetOriginKF()->mnId)
            {
                pMP->SetWorldPos(mProblem.Xw[mvnPoint[i]].cast<float>());
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA = mProblem.Xw[mvnPoint[i]].cast<float>();
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    }

    Digest GlobalBaTask::Input() const
    {
        Digest digest;
        for(std::size_t i = 0; i < mProblem.poses(); i++)
        {
            const Eigen::Quaterniond &R = mProblem.Rcw[i];
            const Eigen::Vector3d &t = mProblem.tcw[i];
            digest.Add('K', mvpPoseKF[i]->mnId, R.x(), R.y(), R.z(), R.w(), t.x(), t.y(), t.z(),
                       mProblem.poseFixed[i] != 0);
        }
        for(std::size_t i = 0; i < mProblem.points(); i++)
        {
            const Eigen::Vector3d &X = mProblem.Xw[i];
            digest.Add('P', mvpPointMP[i]->mnId, X.x(), X.y(), X.z());
        }
        for(std::size_t i = 0; i < mProblem.observations(); i++)
        {
            const Eigen::Vector3d &uv = mProblem.uv[i];
            digest.Add('O', i, static_cast<int>(mProblem.kind[i]), mvpPoseKF[mProblem.pose[i]]->mnId,
                       mvpPointMP[mProblem.point[i]]->mnId, uv.x(), uv.y(), uv.z(), mProblem.invSigma2[i],
                       mProblem.robust[i] != 0);
        }
        return digest;
    }

    bool GlobalBaTask::Matches(const unsigned long nLoopKF) const
    {
        const bool bDirect = nLoopKF == mpMap->GetOriginKF()->mnId;
        for(size_t i = 0; i < mvpKF.size(); i++)
        {
            KeyFrame* pKF = mvpKF[i];
            const int n = mvnPose[i];
            if(pKF->isBad() || n < 0)
                continue;
            if(bDirect)
            {
                const Sophus::SE3f Tcw(mProblem.Rcw[n].cast<float>(), mProblem.tcw[n].cast<float>());
                const Sophus::SE3f Tmap = pKF->GetPose();
                if(std::memcmp(Tcw.data(), Tmap.data(), 7 * sizeof(float)) != 0)
                    return false;
            }
            else
            {
                const Sophus::SE3f Tcw = Sophus::SE3d(mProblem.Rcw[n], mProblem.tcw[n]).cast<float>();
                if(pKF->mnBAGlobalForKF != nLoopKF ||
                   std::memcmp(Tcw.data(), pKF->mTcwGBA.data(), 7 * sizeof(float)) != 0)
                    return false;
            }
        }
        for(size_t i = 0; i < mvpMP.size(); i++)
        {
            MapPoint* pMP = mvpMP[i];
            if(mvnPoint[i] < 0 || pMP->isBad())
                continue;
            const Eigen::Vector3f X = mProblem.Xw[mvnPoint[i]].cast<float>();
            if(bDirect)
            {
                const Eigen::Vector3f Xmap = pMP->GetWorldPos();
                if(std::memcmp(X.data(), Xmap.data(), 3 * sizeof(float)) != 0)
                    return false;
            }
            else if(pMP->mnBAGlobalForKF != nLoopKF ||
                    std::memcmp(X.data(), pMP->mPosGBA.data(), 3 * sizeof(float)) != 0)
                return false;
        }
        return true;
    }

    void Optimizer::BundleAdjustment(const std::vector<KeyFrame*> &vpKFs, const std::vector<MapPoint*> &vpMP,
                                     int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
#else
        GlobalBaTask task;
        task.Build(vpKFs, vpMP, bRobust);

        const std::unique_ptr<optim::BundleAdjuster> pSolver = optim::MakeBundleAdjuster();
        task.Solve(*pSolver, nIterations, pbStopFlag);
        Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

        task.Apply(nLoopKF);
#endif
    }

} // namespace ORB_SLAM3
