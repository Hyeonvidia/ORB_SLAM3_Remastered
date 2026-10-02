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
#include "optimization/InertialBaTask.hpp"
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
    bool sortByVal(const std::pair<MapPoint*, int> &a, const std::pair<MapPoint*, int> &b)
    {
        return (a.second < b.second);
    }

    namespace
    {
        typedef std::vector<std::pair<KeyFrame*, MapPoint*>> ErasedList;

        // The keyframes of an adjustment as they are gathered, then laid out
        // in the order of their ids -- the order v1.0's graph had them in, the
        // id of a pose's vertex being the keyframe's. A keyframe given twice
        // is the one given first, as it was there.
        struct States
        {
            struct Entry
            {
                KeyFrame* pKF;
                bool bFixed;
                bool bInertial;
            };
            std::vector<Entry> vEntries;

            void Add(KeyFrame* pKF, bool bFixed, bool bInertial) { vEntries.push_back({pKF, bFixed, bInertial}); }

            void LayOut(InertialBaWindow &window)
            {
                std::stable_sort(vEntries.begin(), vEntries.end(),
                                 [](const Entry &a, const Entry &b) { return a.pKF->mnId < b.pKF->mnId; });
                for(const Entry &entry : vEntries)
                {
                    KeyFrame* pKFi = entry.pKF;
                    if(window.stateOfId.count(pKFi->mnId))
                        continue;
                    optim::InertialState state;
                    state.pose = BodyPoseOf(pKFi);
                    if(entry.bInertial)
                    {
                        state.velocity = pKFi->GetVelocity().cast<double>();
                        state.gyroBias = pKFi->GetGyroBias().cast<double>();
                        state.accBias = pKFi->GetAccBias().cast<double>();
                    }
                    window.stateOfId[pKFi->mnId] = window.problem.addState(state, entry.bFixed, entry.bInertial);
                    window.stateKF.push_back(pKFi);
                }
            }
        };

        // The state of a keyframe that has a velocity and biases, or -1.
        int InertialStateOf(const InertialBaWindow &window, KeyFrame* pKF)
        {
            const std::map<long unsigned int, int>::const_iterator it = window.stateOfId.find(pKF->mnId);
            if(it == window.stateOfId.end() || !window.problem.inertial[it->second])
                return -1;
            return it->second;
        }

        // The points in the order of their ids, as v1.0's graph had them.
        void LayOutPoints(InertialBaWindow &window, std::vector<MapPoint*> vpMPs)
        {
            std::stable_sort(vpMPs.begin(), vpMPs.end(),
                             [](const MapPoint* a, const MapPoint* b) { return a->mnId < b->mnId; });
            for(MapPoint* pMP : vpMPs)
            {
                if(window.pointOf.count(pMP))
                    continue;
                window.pointOf[pMP] = window.problem.addPoint(pMP->GetWorldPos().cast<double>());
                window.pointMP.push_back(pMP);
            }
        }

        // The observations as they are gathered, point by point in v1.0's
        // order; they are given to the problem once the points have places.
        struct Observations
        {
            struct Entry
            {
                optim::ObservationKind kind;
                KeyFrame* pKF;
                MapPoint* pMP;
                Eigen::Vector3d uv;
                double invSigma2;
                double huber;
            };
            std::vector<Entry> vEntries;

            void Add(optim::ObservationKind kind, KeyFrame* pKF, MapPoint* pMP, const Eigen::Vector3d &uv,
                     double invSigma2, double huber)
            {
                vEntries.push_back({kind, pKF, pMP, uv, invSigma2, huber});
            }

            void GiveTo(InertialBaWindow &window) const
            {
                for(const Entry &entry : vEntries)
                {
                    const std::map<long unsigned int, int>::const_iterator itState = window.stateOfId.find(
                        entry.pKF->mnId);
                    const std::map<MapPoint*, int>::const_iterator itPoint = window.pointOf.find(entry.pMP);
                    if(itState == window.stateOfId.end() || itPoint == window.pointOf.end())
                        continue;
                    window.problem.addObservation(entry.kind, itState->second, itPoint->second, entry.uv,
                                                  entry.invSigma2, entry.huber);
                    window.obsKF.push_back(entry.pKF);
                    window.obsMP.push_back(entry.pMP);
                }
            }
        };

        Sophus::SE3f PoseFound(const InertialBaWindow &window, int n)
        {
            const optim::BodyPose &pose = window.problem.states[n].pose;
            return Sophus::SE3f(pose.Rcw[0].cast<float>(), pose.tcw[0].cast<float>());
        }

        IMU::Bias BiasFound(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba)
        {
            Vector6d b;
            b << bg, ba;
            return IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);
        }

        IMU::Bias BiasFound(const InertialBaWindow &window, int n)
        {
            return BiasFound(window.problem.states[n].gyroBias, window.problem.states[n].accBias);
        }

        bool Same(const Sophus::SE3f &a, const Sophus::SE3f &b)
        {
            return std::memcmp(a.data(), b.data(), 7 * sizeof(float)) == 0;
        }

        bool Same(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
        {
            return std::memcmp(a.data(), b.data(), 3 * sizeof(float)) == 0;
        }

        bool Same(const IMU::Bias &a, const IMU::Bias &b)
        {
            return a.bax == b.bax && a.bay == b.bay && a.baz == b.baz && a.bwx == b.bwx && a.bwy == b.bwy &&
                   a.bwz == b.bwz;
        }

        // Whether a keyframe holds the state found for it.
        bool Holds(const InertialBaWindow &window, KeyFrame* pKFi)
        {
            const int n = window.stateOfId.at(pKFi->mnId);
            if(!Same(PoseFound(window, n), pKFi->GetPose()))
                return false;
            if(pKFi->bImu && window.problem.inertial[n])
            {
                const Eigen::Vector3f v = window.problem.states[n].velocity.cast<float>();
                if(!Same(v, pKFi->GetVelocity()) || !Same(BiasFound(window, n), pKFi->GetImuBias()))
                    return false;
            }
            return true;
        }

        bool HoldsPoints(const InertialBaWindow &window)
        {
            for(std::size_t j = 0; j < window.pointMP.size(); j++)
            {
                const Eigen::Vector3f X = window.problem.Xw[j].cast<float>();
                if(!Same(X, window.pointMP[j]->GetWorldPos()))
                    return false;
            }
            return true;
        }
    } // namespace

    void LocalInertialBaTask::Build(KeyFrame* pKF, const bool bLarge, const bool bRecInit)
    {
        mbLarge = bLarge;
        Map* pCurrentMap = pKF->GetMap();

        int maxOpt = 10;
        int opt_it = 10;
        if(bLarge)
        {
            maxOpt = 25;
            opt_it = 4;
        }
        mnIterations = opt_it;
        const int Nd = std::min((int)pCurrentMap->KeyFramesInMap() - 2, maxOpt);

        std::vector<KeyFrame*> &vpOptimizableKFs = mvpLocalKF;

        vpOptimizableKFs.reserve(Nd);
        vpOptimizableKFs.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;
        mvpMarkedLocal.push_back(pKF);
        for(int i = 1; i < Nd; i++)
        {
            if(vpOptimizableKFs.back()->mPrevKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
                vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
                mvpMarkedLocal.push_back(vpOptimizableKFs.back());
            }
            else
                break;
        }

        int N = vpOptimizableKFs.size();

        // Optimizable points seen by temporal optimizable keyframes
        std::vector<MapPoint*> &lLocalMapPoints = mvpPoints;
        for(int i = 0; i < N; i++)
        {
            std::vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
            for(std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
            }
        }

        // Fixed Keyframe: First frame previous KF to optimization window)
        std::vector<KeyFrame*> &lFixedKeyFrames = mvpFixedKF;
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
            mvpMarkedFixed.push_back(vpOptimizableKFs.back()->mPrevKF);
        }
        else
        {
            vpOptimizableKFs.back()->mnBALocalForKF = 0;
            vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
            mvpMarkedFixed.push_back(vpOptimizableKFs.back());
            lFixedKeyFrames.push_back(vpOptimizableKFs.back());
            vpOptimizableKFs.pop_back();
        }

        // Optimizable visual KFs: none. v1.0 took at most maxCovKF of the
        // covisible keyframes here, and maxCovKF was 0.

        // Fixed KFs which are not covisible optimizable
        const int maxFixKF = 200;

        for(std::vector<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend;
            lit++)
        {
            MapPoint::ObservationMap observations = (*lit)->GetObservations();
            for(MapPoint::ObservationMap::iterator mit = observations.begin(), mend = observations.end(); mit != mend;
                mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    mvpMarkedFixed.push_back(pKFi);
                    if(!pKFi->isBad())
                    {
                        lFixedKeyFrames.push_back(pKFi);
                        break;
                    }
                }
            }
            if(lFixedKeyFrames.size() >= maxFixKF)
                break;
        }

        // Set Local temporal KeyFrame vertices, and the fixed ones
        N = vpOptimizableKFs.size();
        States states;
        for(int i = 0; i < N; i++)
            states.Add(vpOptimizableKFs[i], false, vpOptimizableKFs[i]->bImu);
        // This should be done only for keyframe just before temporal window
        for(KeyFrame* pKFi : lFixedKeyFrames)
            states.Add(pKFi, true, pKFi->bImu);
        states.LayOut(mWindow);

        // Create intertial constraints
        for(int i = 0; i < N; i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            if(!pKFi->mPrevKF)
            {
                std::cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << std::endl;
                continue;
            }
            if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                const int n1 = InertialStateOf(mWindow, pKFi->mPrevKF);
                const int n2 = InertialStateOf(mWindow, pKFi);

                if(n1 < 0 || n2 < 0)
                {
                    std::cerr << "Error: an inertial term between keyframes " << pKFi->mPrevKF->mnId << " and "
                              << pKFi->mnId << ", one of which is not in the window" << std::endl;
                    continue;
                }

                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                const double huber = (i == N - 1 || bRecInit) ? std::sqrt(16.92) : 0.0;
                const double scale = (i == N - 1) ? 1e-2 : 1.0;

                Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
                Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();

                mWindow.preintegrations.emplace_back(pKFi->mpImuPreintegrated);
                mWindow.problem.addTerm(n1, n2, &mWindow.preintegrations.back(), huber, scale, InfoG, InfoA);
            }
            else
                std::cout << "ERROR building inertial edge" << std::endl;
        }

        // Set MapPoint vertices
        LayOutPoints(mWindow, lLocalMapPoints);

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        Observations obs_;
        for(std::vector<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend;
            lit++)
        {
            MapPoint* pMP = *lit;
            const MapPoint::ObservationMap observations = pMP->GetObservations();

            // Create visual constraints
            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(), mend = observations.end();
                mit != mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                    continue;

                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = std::get<0>(mit->second);

                    cv::KeyPoint kpUn;

                    // Monocular left observation
                    if(leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;

                        obs_.Add(optim::kMono, pKFi, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2,
                                 thHuberMono);
                    }
                    // Stereo-observation
                    else if(leftIndex != -1) // Stereo observation
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];

                        const float kp_ur = pKFi->mvuRight[leftIndex];
                        Eigen::Matrix<double, 3, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;

                        obs_.Add(optim::kStereo, pKFi, pMP, obs, invSigma2, thHuberStereo);
                    }

                    // Monocular right observation
                    if(pKFi->mpCamera2)
                    {
                        int rightIndex = std::get<1>(mit->second);

                        if(rightIndex != -1)
                        {
                            rightIndex -= pKFi->NLeft;

                            Eigen::Matrix<double, 2, 1> obs;
                            cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                            obs << kp.pt.x, kp.pt.y;

                            // Add here uncerteinty
                            const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                            // The level is that of the left keypoint, or 0 where
                            // there is none: as v1.0 had it.
                            const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;

                            obs_.Add(optim::kRight, pKFi, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2,
                                     thHuberMono);
                        }
                    }
                }
            }
        }
        obs_.GiveTo(mWindow);
    }

    void LocalInertialBaTask::Solve(optim::InertialBundleAdjuster &solver)
    {
        optim::SolveOptions options;
        options.nIterations = mnIterations; // Originally to 2
        options.damping = optim::SolveOptions::kValue;
        options.dampingValue = mbLarge ? 1e-2 : 1e0; // to avoid iterating for finding optimal lambda
        solver.Solve(mWindow.problem, options);
    }

    const std::vector<std::pair<KeyFrame*, MapPoint*>> &LocalInertialBaTask::Classify()
    {
        const optim::InertialBaProblem &problem = mWindow.problem;
        mvToErase.clear();
        mvToErase.reserve(problem.observations());
        mJudged = Digest();

        const float chi2Mono2 = 5.991;
        const float chi2Stereo2 = 7.815;

        // Check inlier observations
        // Mono, of either camera
        std::size_t nMono = 0;
        for(std::size_t i = 0, iend = problem.observations(); i < iend; i++)
        {
            if(problem.kind[i] == optim::kStereo)
                continue;
            MapPoint* pMP = mWindow.obsMP[i];
            bool bClose = pMP->mTrackDepth < 10.f;
            const bool bBad = pMP->isBad();
            mJudged.Add('M', nMono++, bClose, bBad);

            if(bBad)
                continue;

            if((problem.chi2[i] > chi2Mono2 && !bClose) || (problem.chi2[i] > 1.5f * chi2Mono2 && bClose) ||
               !problem.depthPositive[i])
            {
                KeyFrame* pKFi = mWindow.obsKF[i];
                mvToErase.push_back(std::make_pair(pKFi, pMP));
            }
        }

        // Stereo
        std::size_t nStereo = 0;
        for(std::size_t i = 0, iend = problem.observations(); i < iend; i++)
        {
            if(problem.kind[i] != optim::kStereo)
                continue;
            MapPoint* pMP = mWindow.obsMP[i];
            const bool bBad = pMP->isBad();
            mJudged.Add('S', nStereo++, bBad);

            if(bBad)
                continue;

            if(problem.chi2[i] > chi2Stereo2)
            {
                KeyFrame* pKFi = mWindow.obsKF[i];
                mvToErase.push_back(std::make_pair(pKFi, pMP));
            }
        }
        return mvToErase;
    }

    bool LocalInertialBaTask::Failed() const
    {
        const float err = mWindow.problem.costBefore;
        const float err_end = mWindow.problem.costAfter;

        // TODO: Some convergence problems have been detected here
        return (2 * err < err_end || std::isnan(err) || std::isnan(err_end)) && !mbLarge;
    }

    void LocalInertialBaTask::Apply(Map* pMap)
    {
        Classify();

        // Get Map Mutex and erase outliers
        std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

        if(Failed())
        {
            std::cout << "FAIL LOCAL-INERTIAL BA!!!!" << std::endl;
            return;
        }

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

        for(KeyFrame* pKFi : mvpFixedKF)
            pKFi->mnBAFixedForKF = 0;

        // Recover optimized data
        // Local temporal Keyframes
        for(KeyFrame* pKFi : mvpLocalKF)
        {
            const int n = mWindow.stateOfId.at(pKFi->mnId);
            pKFi->SetPose(PoseFound(mWindow, n));
            pKFi->mnBALocalForKF = 0;

            if(pKFi->bImu)
            {
                pKFi->SetVelocity(mWindow.problem.states[n].velocity.cast<float>());
                pKFi->SetNewBias(BiasFound(mWindow, n));
            }
        }

        //Points
        for(MapPoint* pMP : mvpPoints)
        {
            pMP->SetWorldPos(mWindow.problem.Xw[mWindow.pointOf.at(pMP)].cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    }

    bool LocalInertialBaTask::Matches(const ErasedList &erased, const bool bFailed) const
    {
        if(erased != mvToErase || bFailed != Failed())
            return false;
        if(bFailed)
            return true;
        for(KeyFrame* pKFi : mvpLocalKF)
            if(!Holds(mWindow, pKFi))
                return false;
        return HoldsPoints(mWindow);
    }

    void LocalInertialBaTask::ResetMarks()
    {
        // No keyframe has this id, so v1.0's body takes none of them as seen.
        const long unsigned int none = ~0ul;
        for(KeyFrame* pKFi : mvpMarkedLocal)
            pKFi->mnBALocalForKF = none;
        for(KeyFrame* pKFi : mvpMarkedFixed)
            pKFi->mnBAFixedForKF = none;
        for(MapPoint* pMP : mvpPoints)
            pMP->mnBALocalForKF = none;
    }

    bool FullInertialBaTask::Build(Map* pMap, const bool bFixLocal, const bool bInit, const float priorG,
                                   const float priorA)
    {
        mpMap = pMap;
        mbInit = bInit;
        long unsigned int maxKFid = pMap->GetMaxKFid();
        mvpKF = pMap->GetAllKeyFrames();
        const std::vector<KeyFrame*> &vpKFs = mvpKF;
        const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

        int nNonFixed = 0;

        // Set KeyFrame vertices
        KeyFrame* pIncKF = nullptr;
        States states;
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;
            pIncKF = pKFi;
            bool bFixed = false;
            if(bFixLocal)
            {
                bFixed = (pKFi->mnBALocalForKF >= (maxKFid - 1)) || (pKFi->mnBAFixedForKF >= (maxKFid - 1));
                if(!bFixed)
                    nNonFixed++;
            }
            states.Add(pKFi, bFixed, pKFi->bImu);
        }
        states.LayOut(mWindow);
        mvnState.assign(vpKFs.size(), -1);
        for(size_t i = 0; i < vpKFs.size(); i++)
            if(vpKFs[i]->mnId <= maxKFid)
                mvnState[i] = mWindow.stateOfId.at(vpKFs[i]->mnId);

        optim::InertialBaProblem &problem = mWindow.problem;
        if(bInit)
        {
            problem.sharedBias = true;
            if(pIncKF)
            {
                problem.sharedGyroBias = pIncKF->GetGyroBias().cast<double>();
                problem.sharedAccBias = pIncKF->GetAccBias().cast<double>();
            }
        }

        if(bFixLocal)
        {
            if(nNonFixed < 3)
                return false;
        }

        // IMU links
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(!pKFi->mPrevKF)
            {
                Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
                continue;
            }

            if(pKFi->mPrevKF && pKFi->mnId <= maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                    continue;
                if(pKFi->bImu && pKFi->mPrevKF->bImu)
                {
                    pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                    const int n1 = InertialStateOf(mWindow, pKFi->mPrevKF);
                    const int n2 = InertialStateOf(mWindow, pKFi);

                    if(n1 < 0 || n2 < 0)
                    {
                        std::cout << "Error: an inertial term between keyframes " << pKFi->mPrevKF->mnId << " and "
                                  << pKFi->mnId << ", one of which is not in the map" << std::endl;
                        continue;
                    }

                    Eigen::Matrix3d InfoG = Eigen::Matrix3d::Identity();
                    Eigen::Matrix3d InfoA = Eigen::Matrix3d::Identity();
                    if(!bInit)
                    {
                        InfoG = pKFi->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
                        InfoA = pKFi->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
                    }

                    mWindow.preintegrations.emplace_back(pKFi->mpImuPreintegrated);
                    problem.addTerm(n1, n2, &mWindow.preintegrations.back(), std::sqrt(16.92), 1.0, InfoG, InfoA);
                }
                else
                    std::cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << std::endl;
            }
        }

        if(bInit)
        {
            // Add prior to comon biases
            problem.biasPriors = true;
            problem.accPriorInformation = priorA;
            problem.gyroPriorInformation = priorG;
        }

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        // A point seen only from keyframes that are held is left out, and one
        // seen from none.
        std::vector<MapPoint*> vpIncluded;
        Observations obs_;
        for(size_t i = 0; i < vpMPs.size(); i++)
        {
            MapPoint* pMP = vpMPs[i];

            const MapPoint::ObservationMap observations = pMP->GetObservations();

            bool bAllFixed = true;
            const std::size_t nBefore = obs_.vEntries.size();

            //Set edges
            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(), mend = observations.end();
                mit != mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnId > maxKFid)
                    continue;

                if(!pKFi->isBad())
                {
                    const std::map<long unsigned int, int>::const_iterator itState = mWindow.stateOfId.find(pKFi->mnId);
                    if(itState == mWindow.stateOfId.end())
                        continue; // v1.0 would have read a vertex that is not there
                    const bool bFixed = problem.fixed[itState->second];

                    const int leftIndex = std::get<0>(mit->second);
                    cv::KeyPoint kpUn;

                    if(leftIndex != -1 && pKFi->mvuRight[std::get<0>(mit->second)] < 0) // Monocular observation
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        if(bAllFixed)
                            if(!bFixed)
                                bAllFixed = false;

                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                        obs_.Add(optim::kMono, pKFi, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2,
                                 thHuberMono);
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                    {
                        kpUn = pKFi->mvKeysUn[leftIndex];
                        const float kp_ur = pKFi->mvuRight[leftIndex];
                        Eigen::Matrix<double, 3, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        if(bAllFixed)
                            if(!bFixed)
                                bAllFixed = false;

                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                        obs_.Add(optim::kStereo, pKFi, pMP, obs, invSigma2, thHuberStereo);
                    }

                    if(pKFi->mpCamera2)
                    { // Monocular right observation
                        int rightIndex = std::get<1>(mit->second);

                        if(rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size())
                        {
                            rightIndex -= pKFi->NLeft;

                            Eigen::Matrix<double, 2, 1> obs;
                            kpUn = pKFi->mvKeysRight[rightIndex];
                            obs << kpUn.pt.x, kpUn.pt.y;

                            if(bAllFixed)
                                if(!bFixed)
                                    bAllFixed = false;

                            const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                            obs_.Add(optim::kRight, pKFi, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2,
                                     thHuberMono);
                        }
                    }
                }
            }

            if(bAllFixed)
                obs_.vEntries.resize(nBefore);
            else
                vpIncluded.push_back(pMP);
        }
        LayOutPoints(mWindow, vpIncluded);
        obs_.GiveTo(mWindow);
        return true;
    }

    void FullInertialBaTask::Solve(optim::InertialBundleAdjuster &solver, const int nIterations, bool* pbStopFlag)
    {
        optim::SolveOptions options;
        options.nIterations = nIterations;
        options.pbStop = pbStopFlag;
        options.damping = optim::SolveOptions::kValue;
        options.dampingValue = 1e-5;
        solver.Solve(mWindow.problem, options);
    }

    void FullInertialBaTask::Apply(const unsigned long nLoopId) const
    {
        const optim::InertialBaProblem &problem = mWindow.problem;

        // Recover optimized data
        //Keyframes
        for(size_t i = 0; i < mvpKF.size(); i++)
        {
            KeyFrame* pKFi = mvpKF[i];
            const int n = mvnState[i];
            if(n < 0)
                continue;
            if(nLoopId == 0)
            {
                pKFi->SetPose(PoseFound(mWindow, n));
            }
            else
            {
                pKFi->mTcwGBA = PoseFound(mWindow, n);
                pKFi->mnBAGlobalForKF = nLoopId;
            }
            if(pKFi->bImu)
            {
                if(nLoopId == 0)
                {
                    pKFi->SetVelocity(problem.states[n].velocity.cast<float>());
                }
                else
                {
                    pKFi->mVwbGBA = problem.states[n].velocity.cast<float>();
                }

                const IMU::Bias b = mbInit ? BiasFound(problem.sharedGyroBias, problem.sharedAccBias)
                                           : BiasFound(mWindow, n);
                if(nLoopId == 0)
                {
                    pKFi->SetNewBias(b);
                }
                else
                {
                    pKFi->mBiasGBA = b;
                }
            }
        }

        //Points
        for(std::size_t j = 0; j < mWindow.pointMP.size(); j++)
        {
            MapPoint* pMP = mWindow.pointMP[j];

            if(nLoopId == 0)
            {
                pMP->SetWorldPos(problem.Xw[j].cast<float>());
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA = problem.Xw[j].cast<float>();
                pMP->mnBAGlobalForKF = nLoopId;
            }
        }

        mpMap->IncreaseChangeIndex();
    }

    Digest FullInertialBaTask::Input() const
    {
        const optim::InertialBaProblem &problem = mWindow.problem;
        Digest digest;
        for(std::size_t i = 0; i < problem.states.size(); i++)
        {
            const optim::InertialState &state = problem.states[i];
            const long unsigned int id = mWindow.stateKF[i]->mnId;
            digest.Add('K', id, state.pose.Rcw[0], state.pose.tcw[0], problem.fixed[i] != 0);
            if(problem.inertial[i])
            {
                digest.Add('v', id, state.velocity);
                if(!problem.sharedBias)
                    digest.Add('b', id, state.gyroBias, state.accBias);
            }
        }
        if(problem.sharedBias)
            digest.Add('B', problem.sharedGyroBias, problem.sharedAccBias);
        for(std::size_t k = 0; k < problem.terms(); k++)
        {
            const IMU::Preintegrated* pInt = problem.preintegration[k];
            digest.Add('I', mWindow.stateKF[problem.from[k]]->mnId, mWindow.stateKF[problem.to[k]]->mnId, pInt->dT,
                       pInt->dR, pInt->dV, pInt->dP);
        }
        for(std::size_t j = 0; j < problem.Xw.size(); j++)
            digest.Add('P', mWindow.pointMP[j]->mnId, problem.Xw[j]);
        // An observation's place is among those of its point.
        std::size_t nOfPoint = 0;
        for(std::size_t k = 0; k < problem.observations(); k++)
        {
            if(k > 0 && problem.point[k] != problem.point[k - 1])
                nOfPoint = 0;
            const Eigen::Vector3d &uv = problem.uv[k];
            digest.Add('O', nOfPoint++, static_cast<int>(problem.kind[k]), mWindow.obsKF[k]->mnId,
                       mWindow.obsMP[k]->mnId, uv.x(), uv.y(), uv.z(), problem.invSigma2[k]);
        }
        return digest;
    }

    bool FullInertialBaTask::Matches(const unsigned long nLoopId) const
    {
        const optim::InertialBaProblem &problem = mWindow.problem;
        for(size_t i = 0; i < mvpKF.size(); i++)
        {
            KeyFrame* pKFi = mvpKF[i];
            const int n = mvnState[i];
            if(n < 0)
                continue;
            if(!Same(PoseFound(mWindow, n), nLoopId == 0 ? pKFi->GetPose() : pKFi->mTcwGBA))
                return false;
            if(nLoopId != 0 && pKFi->mnBAGlobalForKF != nLoopId)
                return false;
            if(pKFi->bImu)
            {
                const Eigen::Vector3f v = problem.states[n].velocity.cast<float>();
                if(!Same(v, nLoopId == 0 ? pKFi->GetVelocity() : pKFi->mVwbGBA))
                    return false;
                const IMU::Bias b = mbInit ? BiasFound(problem.sharedGyroBias, problem.sharedAccBias)
                                           : BiasFound(mWindow, n);
                if(!Same(b, nLoopId == 0 ? pKFi->GetImuBias() : pKFi->mBiasGBA))
                    return false;
            }
        }
        for(std::size_t j = 0; j < mWindow.pointMP.size(); j++)
        {
            MapPoint* pMP = mWindow.pointMP[j];
            const Eigen::Vector3f X = problem.Xw[j].cast<float>();
            if(!Same(X, nLoopId == 0 ? pMP->GetWorldPos() : pMP->mPosGBA))
                return false;
            if(nLoopId != 0 && pMP->mnBAGlobalForKF != nLoopId)
                return false;
        }
        return true;
    }

    void MergeInertialBaTask::Build(KeyFrame* pCurrKF, KeyFrame* pMergeKF)
    {
        const int Nd = 6;
        const unsigned long maxKFid = pCurrKF->mnId;

        std::vector<KeyFrame*> &vpOptimizableKFs = mvpLocalKF;
        vpOptimizableKFs.reserve(2 * Nd);

        // For cov KFS, inertial parameters are not optimized
        const int maxCovKF = 30;
        std::vector<KeyFrame*> &vpOptimizableCovKFs = mvpCovKF;
        vpOptimizableCovKFs.reserve(maxCovKF);

        auto MarkLocal = [this, pCurrKF](KeyFrame* pKFi)
        {
            pKFi->mnBALocalForKF = pCurrKF->mnId;
            mvpMarkedLocal.push_back(pKFi);
        };
        auto MarkFixed = [this, pCurrKF](KeyFrame* pKFi)
        {
            pKFi->mnBAFixedForKF = pCurrKF->mnId;
            mvpMarkedFixed.push_back(pKFi);
        };

        // Add sliding window for current KF
        vpOptimizableKFs.push_back(pCurrKF);
        MarkLocal(pCurrKF);
        for(int i = 1; i < Nd; i++)
        {
            if(vpOptimizableKFs.back()->mPrevKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
                MarkLocal(vpOptimizableKFs.back());
            }
            else
                break;
        }

        std::vector<KeyFrame*> lFixedKeyFrames;
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            MarkLocal(vpOptimizableKFs.back()->mPrevKF);
        }
        else
        {
            vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
            vpOptimizableKFs.pop_back();
        }

        // Add temporal neighbours to merge KF (previous and next KFs)
        vpOptimizableKFs.push_back(pMergeKF);
        MarkLocal(pMergeKF);

        // Previous KFs
        for(int i = 1; i < (Nd / 2); i++)
        {
            if(vpOptimizableKFs.back()->mPrevKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
                MarkLocal(vpOptimizableKFs.back());
            }
            else
                break;
        }

        // We fix just once the old map
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
            MarkFixed(vpOptimizableKFs.back()->mPrevKF);
        }
        else
        {
            vpOptimizableKFs.back()->mnBALocalForKF = 0;
            MarkFixed(vpOptimizableKFs.back());
            lFixedKeyFrames.push_back(vpOptimizableKFs.back());
            vpOptimizableKFs.pop_back();
        }

        // Next KFs
        if(pMergeKF->mNextKF)
        {
            vpOptimizableKFs.push_back(pMergeKF->mNextKF);
            MarkLocal(vpOptimizableKFs.back());
        }

        while(vpOptimizableKFs.size() < (2 * Nd))
        {
            if(vpOptimizableKFs.back()->mNextKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
                MarkLocal(vpOptimizableKFs.back());
            }
            else
                break;
        }

        int N = vpOptimizableKFs.size();

        // Optimizable points seen by optimizable keyframes
        std::vector<MapPoint*> &lLocalMapPoints = mvpPoints;
        std::map<MapPoint*, int> mLocalObs;
        for(int i = 0; i < N; i++)
        {
            std::vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
            for(std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                    {
                        if(pMP->mnBALocalForKF != pCurrKF->mnId)
                        {
                            mLocalObs[pMP] = 1;
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pCurrKF->mnId;
                        }
                        else
                        {
                            mLocalObs[pMP]++;
                        }
                    }
            }
        }

        std::vector<std::pair<MapPoint*, int>> pairs;
        pairs.reserve(mLocalObs.size());
        for(auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
            pairs.push_back(*itr);
        std::sort(pairs.begin(), pairs.end(), sortByVal);

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        int i = 0;
        for(std::vector<std::pair<MapPoint*, int>>::iterator lit = pairs.begin(), lend = pairs.end(); lit != lend;
            lit++, i++)
        {
            MapPoint::ObservationMap observations = lit->first->GetObservations();
            if(i >= maxCovKF)
                break;
            for(MapPoint::ObservationMap::iterator mit = observations.begin(), mend = observations.end(); mit != mend;
                mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF != pCurrKF->mnId &&
                   pKFi->mnBAFixedForKF != pCurrKF->mnId) // If optimizable or already included...
                {
                    MarkLocal(pKFi);
                    if(!pKFi->isBad())
                    {
                        vpOptimizableCovKFs.push_back(pKFi);
                        break;
                    }
                }
            }
        }

        // Set Local KeyFrame vertices, the covisible ones, and the fixed one
        N = vpOptimizableKFs.size();
        States states;
        for(int i = 0; i < N; i++)
            states.Add(vpOptimizableKFs[i], false, vpOptimizableKFs[i]->bImu);
        for(KeyFrame* pKFi : vpOptimizableCovKFs)
            states.Add(pKFi, false, pKFi->bImu);
        for(KeyFrame* pKFi : lFixedKeyFrames)
            states.Add(pKFi, true, pKFi->bImu);
        states.LayOut(mWindow);

        // Create intertial constraints
        for(int i = 0; i < N; i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            if(!pKFi->mPrevKF)
            {
                Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
                continue;
            }
            if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                const int n1 = InertialStateOf(mWindow, pKFi->mPrevKF);
                const int n2 = InertialStateOf(mWindow, pKFi);

                if(n1 < 0 || n2 < 0)
                {
                    std::cerr << "Error: an inertial term between keyframes " << pKFi->mPrevKF->mnId << " and "
                              << pKFi->mnId << ", one of which is not in the window" << std::endl;
                    continue;
                }

                Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
                Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();

                mWindow.preintegrations.emplace_back(pKFi->mpImuPreintegrated);
                mWindow.problem.addTerm(n1, n2, &mWindow.preintegrations.back(), std::sqrt(16.92), 1.0, InfoG, InfoA);
            }
            else
                Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
        }

        Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);

        // Set MapPoint vertices
        LayOutPoints(mWindow, lLocalMapPoints);

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        Observations obs_;
        for(std::vector<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend;
            lit++)
        {
            MapPoint* pMP = *lit;
            if(!pMP)
                continue;

            const MapPoint::ObservationMap observations = pMP->GetObservations();

            // Create visual constraints
            for(MapPoint::ObservationMap::const_iterator mit = observations.begin(), mend = observations.end();
                mit != mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(!pKFi)
                    continue;

                if((pKFi->mnBALocalForKF != pCurrKF->mnId) && (pKFi->mnBAFixedForKF != pCurrKF->mnId))
                    continue;

                if(pKFi->mnId > maxKFid)
                {
                    continue;
                }

                if(!mWindow.stateOfId.count(pKFi->mnId))
                    continue;

                if(!pKFi->isBad())
                {
                    // Seen by the second camera only: v1.0 read the keypoint
                    // before the first, whatever was there.
                    if(std::get<0>(mit->second) < 0)
                        continue;

                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[std::get<0>(mit->second)];

                    if(pKFi->mvuRight[std::get<0>(mit->second)] < 0) // Monocular observation
                    {
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                        obs_.Add(optim::kMono, pKFi, pMP, Eigen::Vector3d(obs.x(), obs.y(), 0.0), invSigma2,
                                 thHuberMono);
                    }
                    else // stereo observation
                    {
                        const float kp_ur = pKFi->mvuRight[std::get<0>(mit->second)];
                        Eigen::Matrix<double, 3, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                        obs_.Add(optim::kStereo, pKFi, pMP, obs, invSigma2, thHuberStereo);
                    }
                }
            }
        }
        obs_.GiveTo(mWindow);
    }

    void MergeInertialBaTask::Solve(optim::InertialBundleAdjuster &solver, bool* pbStopFlag)
    {
        optim::SolveOptions options;
        options.nIterations = 8;
        options.pbStop = pbStopFlag;
        options.damping = optim::SolveOptions::kValue;
        options.dampingValue = 1e3;
        solver.Solve(mWindow.problem, options);
    }

    const std::vector<std::pair<KeyFrame*, MapPoint*>> &MergeInertialBaTask::Classify()
    {
        const optim::InertialBaProblem &problem = mWindow.problem;
        mvToErase.clear();
        mvToErase.reserve(problem.observations());

        const float chi2Mono2 = 5.991;
        const float chi2Stereo2 = 7.815;

        // Check inlier observations
        // Mono, then stereo
        for(const bool bStereo : {false, true})
        {
            for(std::size_t i = 0, iend = problem.observations(); i < iend; i++)
            {
                if((problem.kind[i] == optim::kStereo) != bStereo)
                    continue;
                MapPoint* pMP = mWindow.obsMP[i];

                if(pMP->isBad())
                    continue;

                if(problem.chi2[i] > (bStereo ? chi2Stereo2 : chi2Mono2))
                {
                    KeyFrame* pKFi = mWindow.obsKF[i];
                    mvToErase.push_back(std::make_pair(pKFi, pMP));
                }
            }
        }
        return mvToErase;
    }

    void MergeInertialBaTask::Apply(Map* pMap, KeyFrameAndPose &corrPoses)
    {
        Classify();

        // Get Map Mutex and erase outliers
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
        //Keyframes, those of the window and then the covisible ones
        for(const std::vector<KeyFrame*>* pvpKFs : {&mvpLocalKF, &mvpCovKF})
        {
            for(KeyFrame* pKFi : *pvpKFs)
            {
                const int n = mWindow.stateOfId.at(pKFi->mnId);
                pKFi->SetPose(PoseFound(mWindow, n));

                Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
                corrPoses[pKFi] = g2oSiw;

                if(pKFi->bImu)
                {
                    pKFi->SetVelocity(mWindow.problem.states[n].velocity.cast<float>());
                    pKFi->SetNewBias(BiasFound(mWindow, n));
                }
            }
        }

        //Points
        for(MapPoint* pMP : mvpPoints)
        {
            pMP->SetWorldPos(mWindow.problem.Xw[mWindow.pointOf.at(pMP)].cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    }

    bool MergeInertialBaTask::Matches(const ErasedList &erased) const
    {
        if(erased != mvToErase)
            return false;
        for(KeyFrame* pKFi : mvpLocalKF)
            if(!Holds(mWindow, pKFi))
                return false;
        for(KeyFrame* pKFi : mvpCovKF)
            if(!Holds(mWindow, pKFi))
                return false;
        return HoldsPoints(mWindow);
    }

    void MergeInertialBaTask::ResetMarks()
    {
        // No keyframe has this id, so v1.0's body takes none of them as seen.
        const long unsigned int none = ~0ul;
        for(KeyFrame* pKFi : mvpMarkedLocal)
            pKFi->mnBALocalForKF = none;
        for(KeyFrame* pKFi : mvpMarkedFixed)
            pKFi->mnBAFixedForKF = none;
        for(MapPoint* pMP : mvpPoints)
            pMP->mnBALocalForKF = none;
    }

    void Optimizer::FullInertialBA(Map* pMap, int its, const bool bFixLocal, const long unsigned int nLoopId,
                                   bool* pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd* vSingVal,
                                   bool* bHess)
    {
#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::FullInertialBA(pMap, its, bFixLocal, nLoopId, pbStopFlag, bInit, priorG, priorA, vSingVal, bHess);
#else
        FullInertialBaTask task;
        if(!task.Build(pMap, bFixLocal, bInit, priorG, priorA))
            return;

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        const std::unique_ptr<optim::InertialBundleAdjuster> pSolver = optim::MakeInertialBundleAdjuster();
        task.Solve(*pSolver, its, pbStopFlag);
        task.Apply(nLoopId);
#endif
    }

    void Optimizer::LocalInertialBA(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, int &num_fixedKF, int &num_OptKF,
                                    int &num_MPs, int &num_edges, bool bLarge, bool bRecInit)
    {
#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::LocalInertialBA(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges, bLarge, bRecInit);
#else
        // pbStopFlag: v1.0 gave it to the optimiser after the optimisation.
        LocalInertialBaTask task;
        task.Build(pKF, bLarge, bRecInit);
        const std::unique_ptr<optim::InertialBundleAdjuster> pSolver = optim::MakeInertialBundleAdjuster();
        task.Solve(*pSolver);
        task.Apply(pMap);
#endif
    }

    void Optimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool* pbStopFlag, Map* pMap,
                                    KeyFrameAndPose &corrPoses)
    {
#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::MergeInertialBA(pCurrKF, pMergeKF, pbStopFlag, pMap, corrPoses);
#else
        MergeInertialBaTask task;
        task.Build(pCurrKF, pMergeKF);

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        const std::unique_ptr<optim::InertialBundleAdjuster> pSolver = optim::MakeInertialBundleAdjuster();
        task.Solve(*pSolver, pbStopFlag);
        task.Apply(pMap, corrPoses);
#endif
    }

} // namespace ORB_SLAM3
