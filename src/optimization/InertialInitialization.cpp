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
#include "optimization/BodyPoseOf.hpp"
#include "optimization/InertialAlignmentTask.hpp"
#include "tracking/Frame.hpp"

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include "common/Converter.hpp"

#include <mutex>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "optimization/KeyFrameAndPose.hpp"
#include "common/Verbose.hpp"

namespace ORB_SLAM3
{
    void InertialAlignmentTask::Read(Map* pMap, const bool bSetBias)
    {
        long unsigned int maxKFid = pMap->GetMaxKFid();
        mvpKFs = pMap->GetAllKeyFrames();
        const std::vector<KeyFrame*> &vpKFs = mvpKFs;

        // Set KeyFrame vertices (fixed poses and optimizable velocities), in
        // the order of the keyframes' ids: the order v1.0's graph had them in.
        std::vector<std::size_t> vnOrder;
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;
            vnOrder.push_back(i);
        }
        std::stable_sort(vnOrder.begin(), vnOrder.end(),
                         [&vpKFs](std::size_t a, std::size_t b) { return vpKFs[a]->mnId < vpKFs[b]->mnId; });
        mvnPose.assign(vpKFs.size(), -1);
        std::map<long unsigned int, int> mnPoseOfId;
        for(const std::size_t i : vnOrder)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(mnPoseOfId.count(pKFi->mnId))
                continue;
            mvnPose[i] = mProblem.addPose(BodyPoseOf(pKFi), pKFi->GetVelocity().cast<double>());
            mnPoseOfId[pKFi->mnId] = mvnPose[i];
            mvpPoseKF.push_back(pKFi);
        }

        // Biases
        mProblem.gyroBias = vpKFs.front()->GetGyroBias().cast<double>();
        mProblem.accBias = vpKFs.front()->GetAccBias().cast<double>();

        // Graph edges
        // IMU links with gravity and scale
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId <= maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                    continue;
                if(!pKFi->mpImuPreintegrated)
                    std::cout << "Not preintegrated measurement" << std::endl;

                if(bSetBias)
                    pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());

                const std::map<long unsigned int, int>::const_iterator it1 = mnPoseOfId.find(pKFi->mPrevKF->mnId);
                const std::map<long unsigned int, int>::const_iterator it2 = mnPoseOfId.find(pKFi->mnId);
                if(it1 == mnPoseOfId.end() || it2 == mnPoseOfId.end())
                {
                    std::cout << "Error: an inertial term between keyframes " << pKFi->mPrevKF->mnId << " and "
                              << pKFi->mnId << ", one of which is not in the map" << std::endl;

                    continue;
                }
                mPreintegrations.emplace_back(pKFi->mpImuPreintegrated);
                mProblem.addConstraint(it1->second, it2->second, &mPreintegrations.back());
            }
        }
    }

    void InertialAlignmentTask::Solve(optim::InertialAlignmentSolver &solver)
    {
        solver.Solve(mProblem, mOptions);
    }

    void InertialAlignmentTask::Apply() const
    {
        // Recover optimized data
        // Biases
        Eigen::Matrix<double, 6, 1> vb;
        vb << mProblem.gyroBias, mProblem.accBias;
        const Eigen::Vector3d &bg = mProblem.gyroBias;

        IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

        //Keyframes velocities and biases
        const std::vector<KeyFrame*> &vpKFs = mvpKFs;
        const size_t N = vpKFs.size();
        for(size_t i = 0; i < N; i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(mvnPose[i] < 0)
                continue;

            Eigen::Vector3d Vw = mProblem.velocity[mvnPose[i]]; // Velocity is scaled after
            pKFi->SetVelocity(Vw.cast<float>());

            if((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
            {
                pKFi->SetNewBias(b);
                if(pKFi->mpImuPreintegrated)
                    pKFi->mpImuPreintegrated->Reintegrate();
            }
            else
                pKFi->SetNewBias(b);
        }
    }

    // What each of the three holds and how it is solved.
    void InertialAlignmentTask::Build(Map* pMap, const Eigen::Matrix3d &Rwg, const double scale, const bool bMono,
                                      const bool bFixedVel, const float priorG, const float priorA)
    {
        Read(pMap, true);
        optim::InertialAlignmentProblem &problem = mProblem;
        optim::SolveOptions &options = mOptions;
        problem.Rwg = Rwg;
        problem.scale = scale;
        problem.velocitiesFixed = bFixedVel;
        problem.biasesFixed = bFixedVel;
        problem.gravityFixed = false;
        problem.scaleFixed = !bMono; // Fixed for stereo case
        problem.biasPriors = true;
        problem.accPriorInformation = priorA;
        problem.gyroPriorInformation = priorG;

        options.nIterations = 200;
        if(priorG != 0.f)
        {
            options.damping = optim::SolveOptions::kValue;
            options.dampingValue = 1e3;
        }
    }

    void InertialAlignmentTask::Build(Map* pMap, const float priorG, const float priorA)
    {
        Read(pMap, true);
        optim::InertialAlignmentProblem &problem = mProblem;
        optim::SolveOptions &options = mOptions;
        problem.Rwg = Eigen::Matrix3d::Identity();
        problem.scale = 1.0;
        problem.gravityFixed = true;
        problem.scaleFixed = true; // Fixed since scale is obtained from already well initialized map
        problem.biasPriors = true;
        problem.accPriorInformation = priorA;
        problem.gyroPriorInformation = priorG;

        options.nIterations = 200; // Check number of iterations
        options.damping = optim::SolveOptions::kValue;
        options.dampingValue = 1e3;
    }

    void InertialAlignmentTask::Build(Map* pMap, const Eigen::Matrix3d &Rwg, const double scale)
    {
        Read(pMap, false);
        optim::InertialAlignmentProblem &problem = mProblem;
        optim::SolveOptions &options = mOptions;
        problem.Rwg = Rwg;
        problem.scale = scale;
        // all variables are fixed but gravity and scale
        problem.velocitiesFixed = true;
        problem.biasesFixed = true;
        problem.huber = 1.f;

        options.nIterations = 10;
        options.algorithm = optim::SolveOptions::kGaussNewton;
    }

    void Optimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg,
                                         Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd &covInertial, bool bFixedVel,
                                         bool bGauss, float priorG, float priorA)
    {
        Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
        InertialAlignmentTask task;
        task.Build(pMap, Rwg, scale, bMono, bFixedVel, priorG, priorA);
        const std::unique_ptr<optim::InertialAlignmentSolver> pSolver = optim::MakeInertialAlignmentSolver();
        task.Solve(*pSolver);

        const optim::InertialAlignmentProblem &found = task.Problem();
        bg = found.gyroBias;
        ba = found.accBias;
        scale = found.scale;
        Rwg = found.Rwg;
        task.Apply();
    }

    void Optimizer::InertialOptimization(Map* pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG,
                                         float priorA)
    {
        InertialAlignmentTask task;
        task.Build(pMap, priorG, priorA);
        const std::unique_ptr<optim::InertialAlignmentSolver> pSolver = optim::MakeInertialAlignmentSolver();
        task.Solve(*pSolver);

        const optim::InertialAlignmentProblem &found = task.Problem();
        bg = found.gyroBias;
        ba = found.accBias;
        task.Apply();
    }

    void Optimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d &Rwg, double &scale)
    {
        InertialAlignmentTask task;
        task.Build(pMap, Rwg, scale);
        const std::unique_ptr<optim::InertialAlignmentSolver> pSolver = optim::MakeInertialAlignmentSolver();
        task.Solve(*pSolver);

        // Recover optimized data
        const optim::InertialAlignmentProblem &found = task.Problem();
        scale = found.scale;
        Rwg = found.Rwg;
    }

} // namespace ORB_SLAM3
