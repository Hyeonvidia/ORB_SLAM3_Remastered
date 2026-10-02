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
#include "optimization/PoseTask.hpp"
#include "tracking/Frame.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>

namespace ORB_SLAM3
{

    void PoseTask::Build(Frame* pFrame)
    {
        const float deltaMono = std::sqrt(5.991);
        const float deltaStereo = std::sqrt(7.815);

        const Sophus::SE3<float> Tcw = pFrame->GetPose();
        mProblem.Rcw = Tcw.unit_quaternion().cast<double>();
        mProblem.tcw = Tcw.translation().cast<double>();

        mProblem.rig.camera = pFrame->mpCamera;
        mProblem.rig.camera2 = pFrame->mpCamera2;
        if(pFrame->mpCamera2)
        {
            mProblem.rig.Rrl = pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>();
            mProblem.rig.trl = pFrame->GetRelativePoseTrl().translation().cast<double>();
        }
        mProblem.rig.fx = pFrame->fx;
        mProblem.rig.fy = pFrame->fy;
        mProblem.rig.cx = pFrame->cx;
        mProblem.rig.cy = pFrame->cy;
        mProblem.rig.bf = pFrame->mbf;

        const int N = pFrame->N;
        mProblem.reserve(N);
        mvnFeature.reserve(N);

        for(int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(!pMP)
                continue;

            optim::ObservationKind kind;
            cv::KeyPoint kp;
            float uR = 0.f;
            //Conventional SLAM
            if(!pFrame->mpCamera2)
            {
                kp = pFrame->mvKeysUn[i];
                if(pFrame->mvuRight[i] < 0)
                    kind = optim::kMono;
                else
                {
                    kind = optim::kStereo;
                    uR = pFrame->mvuRight[i];
                }
            }
            //SLAM with respect a rigid body
            else if(i < pFrame->Nleft)
            {
                kp = pFrame->mvKeys[i];
                kind = optim::kMono;
            }
            else
            {
                kp = pFrame->mvKeysRight[i - pFrame->Nleft];
                kind = optim::kRight;
            }

            const float invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
            mProblem.add(kind, pMP->GetWorldPos().cast<double>(), Eigen::Vector3d(kp.pt.x, kp.pt.y, uR), invSigma2,
                         kind == optim::kStereo ? deltaStereo : deltaMono);
            mvnFeature.push_back(i);
        }
        mvbOutlier.assign(mProblem.size(), 0);
    }

    void PoseTask::Solve(optim::PoseSolver &solver)
    {
        const std::size_t n = mProblem.size();
        if(n < 3)
            return;
        mbSolved = true;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        const Eigen::Quaterniond Rcw = mProblem.Rcw;
        const Eigen::Vector3d tcw = mProblem.tcw;

        solver.Prepare(mProblem);
        for(std::size_t it = 0; it < 4; it++)
        {
            mProblem.Rcw = Rcw;
            mProblem.tcw = tcw;
            solver.Solve(mProblem, its[it]);

            mnBad = 0;
            for(std::size_t i = 0; i < n; i++)
            {
                const float chi2 = mProblem.chi2[i];
                const float th = mProblem.kind[i] == optim::kStereo ? chi2Stereo[it] : chi2Mono[it];
                const bool bBad = chi2 > th;
                mvbOutlier[i] = bBad;
                mProblem.active[i] = !bBad;
                if(bBad)
                    mnBad++;

                if(it == 2)
                    mProblem.robust[i] = 0;
            }

            if(n < 10)
                break;
        }
    }

    int PoseTask::Apply(Frame* pFrame) const
    {
        const std::size_t n = mProblem.size();
        for(std::size_t i = 0; i < n; i++)
            pFrame->mvbOutlier[mvnFeature[i]] = mvbOutlier[i];

        if(!mbSolved)
            return 0;

        // Recover optimized pose and return number of inliers
        Sophus::SE3<float> pose(mProblem.Rcw.cast<float>(), mProblem.tcw.cast<float>());
        pFrame->SetPose(pose);

        return static_cast<int>(n) - mnBad;
    }

    int Optimizer::PoseOptimization(Frame* pFrame)
    {
        PoseTask task;
        {
            std::lock_guard<std::mutex> lock(MapPoint::mGlobalMutex);
            task.Build(pFrame);
        }
        const std::unique_ptr<optim::PoseSolver> pSolver = optim::MakePoseSolver();
        task.Solve(*pSolver);
        return task.Apply(pFrame);
    }

} // namespace ORB_SLAM3
