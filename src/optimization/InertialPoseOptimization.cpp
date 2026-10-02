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
#include "optimization/ConstraintPoseImu.hpp"
#include "optimization/InertialPoseTask.hpp"
#include "tracking/Frame.hpp"

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include "common/Converter.hpp"

#include <mutex>

#include <algorithm>
#include <cmath>
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
    Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
    {
        // Goal
        // a  | ab | ac       a*  | 0 | ac*
        // ba | b  | bc  -->  0   | 0 | 0
        // ca | cb | c        ca* | 0 | c*

        // Size of block before block to marginalize
        const int a = start;
        // Size of block to marginalize
        const int b = end - start + 1;
        // Size of block after block to marginalize
        const int c = H.cols() - (end + 1);

        // Reorder as follows:
        // a  | ab | ac       a  | ac | ab
        // ba | b  | bc  -->  ca | c  | cb
        // ca | cb | c        ba | bc | b

        Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(), H.cols());
        if(a > 0)
        {
            Hn.block(0, 0, a, a) = H.block(0, 0, a, a);
            Hn.block(0, a + c, a, b) = H.block(0, a, a, b);
            Hn.block(a + c, 0, b, a) = H.block(a, 0, b, a);
        }
        if(a > 0 && c > 0)
        {
            Hn.block(0, a, a, c) = H.block(0, a + b, a, c);
            Hn.block(a, 0, c, a) = H.block(a + b, 0, c, a);
        }
        if(c > 0)
        {
            Hn.block(a, a, c, c) = H.block(a + b, a + b, c, c);
            Hn.block(a, a + c, c, b) = H.block(a + b, a, c, b);
            Hn.block(a + c, a, b, c) = H.block(a, a + b, b, c);
        }
        Hn.block(a + c, a + c, b, b) = H.block(a, a, b, b);

        // Perform marginalization (Schur complement)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a + c, a + c, b, b), Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues();
        for(int i = 0; i < b; ++i)
        {
            if(singularValues_inv(i) > 1e-6)
                singularValues_inv(i) = 1.0 / singularValues_inv(i);
            else
                singularValues_inv(i) = 0;
        }
        Eigen::MatrixXd invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
        Hn.block(0, 0, a + c, a + c) = Hn.block(0, 0, a + c, a + c) -
                                       Hn.block(0, a + c, a + c, b) * invHb * Hn.block(a + c, 0, b, a + c);
        Hn.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
        Hn.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
        Hn.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

        // Inverse reorder
        // a*  | ac* | 0       a*  | 0 | ac*
        // ca* | c*  | 0  -->  0   | 0 | 0
        // 0   | 0   | 0       ca* | 0 | c*
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(), H.cols());
        if(a > 0)
        {
            res.block(0, 0, a, a) = Hn.block(0, 0, a, a);
            res.block(0, a, a, b) = Hn.block(0, a + c, a, b);
            res.block(a, 0, b, a) = Hn.block(a + c, 0, b, a);
        }
        if(a > 0 && c > 0)
        {
            res.block(0, a + b, a, c) = Hn.block(0, a, a, c);
            res.block(a + b, 0, c, a) = Hn.block(a, 0, c, a);
        }
        if(c > 0)
        {
            res.block(a + b, a + b, c, c) = Hn.block(a, a, c, c);
            res.block(a + b, a, c, b) = Hn.block(a, a + c, c, b);
            res.block(a, a + b, b, c) = Hn.block(a + c, a, b, c);
        }

        res.block(a, a, b, b) = Hn.block(a + c, a + c, b, b);

        return res;
    }

    void InertialPoseTask::Build(Frame* pFrame, const Previous previous)
    {
        mPrevious = previous;

        // Set Frame vertex
        optim::InertialState &state = mProblem.state;
        state.pose = BodyPoseOf(pFrame);
        state.velocity = pFrame->GetVelocity().cast<double>();
        state.gyroBias << pFrame->mImuBias.bwx, pFrame->mImuBias.bwy, pFrame->mImuBias.bwz;
        state.accBias << pFrame->mImuBias.bax, pFrame->mImuBias.bay, pFrame->mImuBias.baz;

        // Set MapPoint vertices
        const int N = pFrame->N;
        const int Nleft = pFrame->Nleft;
        const bool bRight = (Nleft != -1);

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        for(int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;

                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;

                    Add(optim::kMono, i, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2, thHuberMono);
                }
                // Stereo observation
                else if(!bRight)
                {
                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;

                    Add(optim::kStereo, i, pMP, obs, invSigma2, thHuberStereo);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;

                    Add(optim::kRight, i, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2, thHuberMono);
                }
            }
        }
        mvbOutlier.assign(mProblem.size(), 0);

        optim::InertialState &before = mProblem.previous;
        if(previous == kLastKeyFrame)
        {
            KeyFrame* pKF = pFrame->mpLastKeyFrame;
            before.pose = BodyPoseOf(pKF);
            before.velocity = pKF->GetVelocity().cast<double>();
            before.gyroBias = pKF->GetGyroBias().cast<double>();
            before.accBias = pKF->GetAccBias().cast<double>();
            mProblem.previousFixed = true;

            mPreintegration.CopyFrom(pFrame->mpImuPreintegrated);
        }
        else
        {
            // Set Previous Frame Vertex
            Frame* pFp = pFrame->mpPrevFrame;
            before.pose = BodyPoseOf(pFp);
            before.velocity = pFp->GetVelocity().cast<double>();
            before.gyroBias << pFp->mImuBias.bwx, pFp->mImuBias.bwy, pFp->mImuBias.bwz;
            before.accBias << pFp->mImuBias.bax, pFp->mImuBias.bay, pFp->mImuBias.baz;
            mProblem.previousFixed = false;

            mPreintegration.CopyFrom(pFrame->mpImuPreintegratedFrame);

            if(!pFp->mpcpi)
                Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + std::to_string(pFp->mnId),
                                   Verbose::VERBOSITY_NORMAL);

            const ConstraintPoseImu* pPrior = pFp->mpcpi;
            mProblem.priorRwb = pPrior->Rwb;
            mProblem.priorTwb = pPrior->twb;
            mProblem.priorVelocity = pPrior->vwb;
            mProblem.priorGyroBias = pPrior->bg;
            mProblem.priorAccBias = pPrior->ba;
            mProblem.priorInformation = pPrior->H;
            mProblem.priorHuber = 5;
        }
        mProblem.preintegration = &mPreintegration;

        // The walks are of the interval since the last keyframe, whichever
        // state the frame is tied to.
        mProblem.gyroWalkInformation = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
        mProblem.accWalkInformation = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
    }

    void InertialPoseTask::Add(const optim::ObservationKind kind, const int nFeature, MapPoint* pMP,
                               const Eigen::Vector3d &obs, const float invSigma2, const float thHuber)
    {
        mProblem.add(kind, pMP->GetWorldPos().cast<double>(), obs, invSigma2, thHuber);
        mvnFeature.push_back(nFeature);
        mvbClose.push_back(pMP->mTrackDepth < 10.f);
    }

    void InertialPoseTask::Solve(optim::InertialPoseSolver &solver, const bool bRecInit)
    {
        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2MonoKeyFrame[4] = {12, 7.5, 5.991, 5.991};
        const float chi2MonoFrame[4] = {5.991, 5.991, 5.991, 5.991};
        const float* chi2Mono = mPrevious == kLastKeyFrame ? chi2MonoKeyFrame : chi2MonoFrame;
        const float chi2Stereo[4] = {15.6, 9.8, 7.815, 7.815};

        const int its[4] = {10, 10, 10, 10};

        const std::size_t n = mProblem.size();
        // Every term there is, in or out: the observations, the inertial one,
        // the two walks and, for a frame, the prior.
        const std::size_t nEdges = n + (mPrevious == kLastKeyFrame ? 3 : 4);

        solver.Prepare(mProblem);

        int nInliers = 0;
        for(size_t it = 0; it < 4; it++)
        {
            solver.Solve(mProblem, its[it]);

            mnBad = 0;
            nInliers = 0;
            float chi2close = 1.5 * chi2Mono[it];

            for(std::size_t i = 0; i < n; i++)
            {
                const float chi2 = mProblem.chi2[i];

                bool bOut;
                if(mProblem.kind[i] != optim::kStereo)
                {
                    bool bClose = mvbClose[i];
                    bOut = (chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) ||
                           !mProblem.depthPositive[i];
                }
                else
                    bOut = chi2 > chi2Stereo[it];

                if(bOut)
                {
                    mvbOutlier[i] = 1;
                    mProblem.active[i] = 0; // not included in next optimization
                    mnBad++;
                }
                else
                {
                    mvbOutlier[i] = 0;
                    mProblem.active[i] = 1;
                    nInliers++;
                }

                if(it == 2)
                    mProblem.robust[i] = 0;
            }

            if(nEdges < 10)
            {
                break;
            }
        }

        // If not too much tracks, recover not too bad points
        if((nInliers < 30) && !bRecInit)
        {
            mnBad = 0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            solver.Evaluate(mProblem);
            for(std::size_t i = 0; i < n; i++)
            {
                if(mProblem.chi2[i] < (mProblem.kind[i] != optim::kStereo ? chi2MonoOut : chi2StereoOut))
                    mvbOutlier[i] = 0;
                else
                    mnBad++;
            }
        }

        // What the observations that fit and the inertial terms say of the
        // states, for the prior of the next frame.
        std::vector<unsigned char> vbInlier(n);
        for(std::size_t i = 0; i < n; i++)
            vbInlier[i] = !mvbOutlier[i];
        solver.Information(mProblem, vbInlier);
    }

    int InertialPoseTask::Apply(Frame* pFrame) const
    {
        for(std::size_t i = 0; i < mvbOutlier.size(); i++)
            pFrame->mvbOutlier[mvnFeature[i]] = mvbOutlier[i] != 0;

        // Recover optimized pose, velocity and biases
        const optim::InertialState &found = mProblem.state;
        pFrame->SetImuPoseVelocity(found.pose.Rwb.cast<float>(), found.pose.twb.cast<float>(),
                                   found.velocity.cast<float>());
        Eigen::Matrix<double, 6, 1> b;
        b << found.gyroBias, found.accBias;
        pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

        if(mPrevious == kLastKeyFrame)
        {
            // The keyframe was held: what is known of the frame is what the
            // terms say of it.
            Eigen::Matrix<double, 15, 15> H = mProblem.information;

            pFrame->mpcpi = new ConstraintPoseImu(found.pose.Rwb, found.pose.twb, found.velocity, found.gyroBias,
                                                  found.accBias, H);
        }
        else
        {
            // Marginalize previous frame states and generate new prior for frame
            Eigen::Matrix<double, 30, 30> H = mProblem.information;

            H = Optimizer::Marginalize(H, 0, 14);

            pFrame->mpcpi = new ConstraintPoseImu(found.pose.Rwb, found.pose.twb, found.velocity, found.gyroBias,
                                                  found.accBias, H.block<15, 15>(15, 15));
            Frame* pFp = pFrame->mpPrevFrame;
            delete pFp->mpcpi;
            pFp->mpcpi = NULL;
        }

        return static_cast<int>(mProblem.size()) - mnBad;
    }

    int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit)
    {
        InertialPoseTask task;
        {
            std::lock_guard<std::mutex> lock(MapPoint::mGlobalMutex);
            task.Build(pFrame, InertialPoseTask::kLastKeyFrame);
        }
        const std::unique_ptr<optim::InertialPoseSolver> pSolver = optim::MakeInertialPoseSolver();
        task.Solve(*pSolver, bRecInit);
        return task.Apply(pFrame);
    }

    int Optimizer::PoseInertialOptimizationLastFrame(Frame* pFrame, bool bRecInit)
    {
        InertialPoseTask task;
        {
            std::lock_guard<std::mutex> lock(MapPoint::mGlobalMutex);
            task.Build(pFrame, InertialPoseTask::kLastFrame);
        }
        const std::unique_ptr<optim::InertialPoseSolver> pSolver = optim::MakeInertialPoseSolver();
        task.Solve(*pSolver, bRecInit);
        return task.Apply(pFrame);
    }

} // namespace ORB_SLAM3
