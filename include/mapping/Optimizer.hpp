/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
* Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
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


#pragma once

#include "IOptimizer.hpp"
#include <memory>

namespace ORB_SLAM3
{

class LoopClosing;

class Optimizer
{
public:
    // Backend management
    static void SetBackend(std::unique_ptr<IOptimizer> backend);
    static IOptimizer& Get();

    // Static facade delegates — all forward to the active IOptimizer backend

    static void BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag = nullptr, unsigned long nLoopKF = 0,
                                 bool bRobust = true) {
        Get().BundleAdjustment(vpKF, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    static void GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5, bool *pbStopFlag = nullptr,
                                       unsigned long nLoopKF = 0, bool bRobust = true) {
        Get().GlobalBundleAdjustemnt(pMap, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    static void FullInertialBA(Map *pMap, int its, bool bFixLocal = false,
                               unsigned long nLoopKF = 0, bool *pbStopFlag = nullptr,
                               bool bInit = false, float priorG = 1e2, float priorA = 1e6,
                               Eigen::VectorXd *vSingVal = nullptr, bool *bHess = nullptr) {
        Get().FullInertialBA(pMap, its, bFixLocal, nLoopKF, pbStopFlag, bInit, priorG, priorA, vSingVal, bHess);
    }

    static void LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap,
                                      int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges) {
        Get().LocalBundleAdjustment(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges);
    }

    static int PoseOptimization(Frame* pFrame) {
        return Get().PoseOptimization(pFrame);
    }

    static int PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit = false) {
        return Get().PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);
    }

    static int PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit = false) {
        return Get().PoseInertialOptimizationLastFrame(pFrame, bRecInit);
    }

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    static void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const KeyFrameAndPose &NonCorrectedSim3,
                                       const KeyFrameAndPose &CorrectedSim3,
                                       const std::map<KeyFrame *, std::set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale) {
        Get().OptimizeEssentialGraph(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale);
    }

    static void OptimizeEssentialGraph(KeyFrame* pCurKF, std::vector<KeyFrame*> &vpFixedKFs,
                                       std::vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       std::vector<KeyFrame*> &vpNonFixedKFs,
                                       std::vector<MapPoint*> &vpNonCorrectedMPs) {
        Get().OptimizeEssentialGraph(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs);
    }

    // For inertial loopclosing
    static void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                           const KeyFrameAndPose &NonCorrectedSim3,
                                           const KeyFrameAndPose &CorrectedSim3,
                                           const std::map<KeyFrame *, std::set<KeyFrame *> > &LoopConnections) {
        Get().OptimizeEssentialGraph4DoF(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono) (NEW)
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                            Sim3Type &g2oS12, float th2, bool bFixScale,
                            Eigen::Matrix<double,7,7> &mAcumHessian, bool bAllPoints = false) {
        return Get().OptimizeSim3(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, mAcumHessian, bAllPoints);
    }

    // For inertial systems
    static void LocalInertialBA(KeyFrame* pKF, bool *pbStopFlag, Map *pMap,
                                int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges,
                                bool bLarge = false, bool bRecInit = false) {
        Get().LocalInertialBA(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges, bLarge, bRecInit);
    }

    static void MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag,
                                Map *pMap, KeyFrameAndPose &corrPoses) {
        Get().MergeInertialBA(pCurrKF, pMergeKF, pbStopFlag, pMap, corrPoses);
    }

    // Local BA in welding area when two maps are merged
    static void LocalBundleAdjustment(KeyFrame* pMainKF, std::vector<KeyFrame*> vpAdjustKF,
                                      std::vector<KeyFrame*> vpFixedKF, bool *pbStopFlag) {
        Get().LocalBundleAdjustment(pMainKF, vpAdjustKF, vpFixedKF, pbStopFlag);
    }

    // Marginalize block element (start:end,start:end). Perform Schur complement.
    // Marginalized elements are filled with zeros.
    // Pure linear algebra (Schur complement) — not backend-specific.
    static Eigen::MatrixXd Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end);

    // Inertial pose-graph
    static void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale,
                                     Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono,
                                     Eigen::MatrixXd &covInertial, bool bFixedVel = false,
                                     bool bGauss = false, float priorG = 1e2, float priorA = 1e6) {
        Get().InertialOptimization(pMap, Rwg, scale, bg, ba, bMono, covInertial, bFixedVel, bGauss, priorG, priorA);
    }

    static void InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba,
                                     float priorG = 1e2, float priorA = 1e6) {
        Get().InertialOptimization(pMap, bg, ba, priorG, priorA);
    }

    static void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale) {
        Get().InertialOptimization(pMap, Rwg, scale);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
    static std::unique_ptr<IOptimizer> sBackend_;
};

} //namespace ORB_SLAM3
