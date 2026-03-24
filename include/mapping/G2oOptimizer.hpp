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

#include "Map.hpp"
#include "MapPoint.hpp"
#include "KeyFrame.hpp"
#include "core/Types.hpp"
#include "Frame.hpp"

#include <math.h>

#include "g2o/types/types_seven_dof_expmap.h"
#include "g2o/core/sparse_block_matrix.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"

namespace ORB_SLAM3
{

class LoopClosing;

class G2oOptimizer : public IOptimizer
{
public:

    void BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                          int nIterations = 5, bool *pbStopFlag = nullptr, unsigned long nLoopKF = 0,
                          bool bRobust = true) override;

    void GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5, bool *pbStopFlag = nullptr,
                                unsigned long nLoopKF = 0, bool bRobust = true) override;

    void FullInertialBA(Map *pMap, int its, bool bFixLocal = false,
                        unsigned long nLoopKF = 0, bool *pbStopFlag = nullptr,
                        bool bInit = false, float priorG = 1e2, float priorA = 1e6,
                        Eigen::VectorXd *vSingVal = nullptr, bool *bHess = nullptr) override;

    void LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap,
                               int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges) override;

    int PoseOptimization(Frame* pFrame) override;
    int PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit = false) override;
    int PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit = false) override;

    void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                const KeyFrameAndPose &NonCorrectedSim3,
                                const KeyFrameAndPose &CorrectedSim3,
                                const std::map<KeyFrame *, std::set<KeyFrame *> > &LoopConnections,
                                const bool &bFixScale) override;

    void OptimizeEssentialGraph(KeyFrame* pCurKF, std::vector<KeyFrame*> &vpFixedKFs,
                                std::vector<KeyFrame*> &vpFixedCorrectedKFs,
                                std::vector<KeyFrame*> &vpNonFixedKFs,
                                std::vector<MapPoint*> &vpNonCorrectedMPs) override;

    void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                    const KeyFrameAndPose &NonCorrectedSim3,
                                    const KeyFrameAndPose &CorrectedSim3,
                                    const std::map<KeyFrame *, std::set<KeyFrame *> > &LoopConnections) override;

    int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                     Sim3Type &g2oS12, float th2, bool bFixScale,
                     Eigen::Matrix<double,7,7> &mAcumHessian, bool bAllPoints = false) override;

    void LocalInertialBA(KeyFrame* pKF, bool *pbStopFlag, Map *pMap,
                         int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges,
                         bool bLarge = false, bool bRecInit = false) override;

    void MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag,
                         Map *pMap, KeyFrameAndPose &corrPoses) override;

    void LocalBundleAdjustment(KeyFrame* pMainKF, std::vector<KeyFrame*> vpAdjustKF,
                               std::vector<KeyFrame*> vpFixedKF, bool *pbStopFlag) override;

    void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale,
                              Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono,
                              Eigen::MatrixXd &covInertial, bool bFixedVel = false,
                              bool bGauss = false, float priorG = 1e2, float priorA = 1e6) override;

    void InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba,
                              float priorG = 1e2, float priorA = 1e6) override;

    void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale) override;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

} //namespace ORB_SLAM3
