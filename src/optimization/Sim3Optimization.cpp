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
#include "optimization/Sim3Task.hpp"
#include "tracking/Frame.hpp"

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include "common/Converter.hpp"

#include <mutex>

#include <algorithm>
#include <cstddef>
#include <memory>
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
    void Sim3Task::Build(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatches1, const Sim3 &g2oS12,
                         const float th2, const bool bFixScale, const bool bAllPoints)
    {
        mTh2 = th2;

        // Camera poses
        const Eigen::Matrix3f R1w = pKF1->GetRotation();
        const Eigen::Vector3f t1w = pKF1->GetTranslation();
        const Eigen::Matrix3f R2w = pKF2->GetRotation();
        const Eigen::Vector3f t2w = pKF2->GetTranslation();

        mProblem.R12 = g2oS12.rotation();
        mProblem.t12 = g2oS12.translation();
        mProblem.s12 = g2oS12.scale();
        mProblem.fixScale = bFixScale;
        mProblem.camera1 = pKF1->mpCamera;
        mProblem.camera2 = pKF2->mpCamera;

        const float deltaHuber = std::sqrt(th2);
        mProblem.huber = deltaHuber;

        const int N = vpMatches1.size();
        const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();

        for(int i = 0; i < N; i++)
        {
            if(!vpMatches1[i])
                continue;

            MapPoint* pMP1 = vpMapPoints1[i];
            MapPoint* pMP2 = vpMatches1[i];

            const int i2 = std::get<0>(pMP2->GetIndexInKeyFrame(pKF2));

            //TODO The 3D position in KF1 doesn't exist
            if(!pMP1)
                continue;
            if(pMP1->isBad() || pMP2->isBad())
                continue;

            Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
            const Eigen::Vector3f P3D1c = R1w * P3D1w + t1w;
            Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
            const Eigen::Vector3f P3D2c = R2w * P3D2w + t2w;

            if(i2 < 0 && !bAllPoints)
            {
                Verbose::PrintMess("    Remove point -> i2: " + std::to_string(i2) +
                                       "; bAllPoints: " + std::to_string(bAllPoints),
                                   Verbose::VERBOSITY_DEBUG);
                continue;
            }

            if(P3D2c(2) < 0)
            {
                Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
                continue;
            }

            // Set edge x1 = S12*X2
            Eigen::Matrix<double, 2, 1> obs1;
            const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
            obs1 << kpUn1.pt.x, kpUn1.pt.y;
            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];

            // Set edge x2 = S21*X1
            Eigen::Matrix<double, 2, 1> obs2;
            cv::KeyPoint kpUn2;
            if(i2 >= 0)
            {
                kpUn2 = pKF2->mvKeysUn[i2];
                obs2 << kpUn2.pt.x, kpUn2.pt.y;
            }
            else
            {
                float invz = 1 / P3D2c(2);
                float x = P3D2c(0) * invz;
                float y = P3D2c(1) * invz;

                obs2 << x, y;
                kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel);
            }
            float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];

            mProblem.add(P3D1c.cast<double>(), P3D2c.cast<double>(), obs1, obs2, invSigmaSquare1, invSigmaSquare2);
            mvnMatch.push_back(i);
        }
        mvbOut.assign(mProblem.size(), 0);
    }

    void Sim3Task::Solve(optim::Sim3Solver &solver)
    {
        const std::size_t n = mProblem.size();

        // Optimize!
        solver.Prepare(mProblem);
        solver.Solve(mProblem, 5);

        // Check inliers
        int nBad = 0;
        for(std::size_t i = 0; i < n; i++)
        {
            if(mProblem.chi2_1[i] > mTh2 || mProblem.chi2_2[i] > mTh2)
            {
                mvbOut[i] = 1;
                mProblem.active[i] = 0;
                nBad++;
                continue;
            }

            //Check if remove the robust adjustment improve the result
            mProblem.robust[i] = 0;
        }

        int nMoreIterations;
        if(nBad > 0)
            nMoreIterations = 10;
        else
            nMoreIterations = 5;

        if(static_cast<int>(n) - nBad < 10)
            return;

        // Optimize again only with inliers
        solver.Solve(mProblem, nMoreIterations);
        solver.Evaluate(mProblem);

        mnIn = 0;
        for(std::size_t i = 0; i < n; i++)
        {
            if(!mProblem.active[i])
                continue;

            if(mProblem.chi2_1[i] > mTh2 || mProblem.chi2_2[i] > mTh2)
                mvbOut[i] = 1;
            else
                mnIn++;
        }
        mbFound = true;
    }

    int Sim3Task::Apply(std::vector<MapPoint*> &vpMatches1, Sim3 &g2oS12,
                        Eigen::Matrix<double, 7, 7> &mAcumHessian) const
    {
        for(std::size_t i = 0; i < mvbOut.size(); i++)
            if(mvbOut[i])
                vpMatches1[mvnMatch[i]] = static_cast<MapPoint*>(NULL);

        if(!mbFound)
            return 0;

        mAcumHessian = Eigen::MatrixXd::Zero(7, 7);

        // Recover optimized Sim3
        g2oS12.rotation() = mProblem.R12;
        g2oS12.translation() = mProblem.t12;
        g2oS12.scale() = mProblem.s12;

        return mnIn;
    }

    int Optimizer::OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches1, Sim3 &g2oS12,
                                const float th2, const bool bFixScale, Eigen::Matrix<double, 7, 7> &mAcumHessian,
                                const bool bAllPoints)
    {
        Sim3Task task;
        task.Build(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, bAllPoints);
        const std::unique_ptr<optim::Sim3Solver> pSolver = optim::MakeSim3Solver();
        task.Solve(*pSolver);
        return task.Apply(vpMatches1, g2oS12, mAcumHessian);
    }

} // namespace ORB_SLAM3
