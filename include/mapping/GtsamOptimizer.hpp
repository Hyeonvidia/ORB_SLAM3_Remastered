/**
* This file is part of ORB-SLAM3
*
* GTSAM optimizer backend implementing IOptimizer interface.
* All GTSAM headers are kept out of this header (included only in the .cpp).
*/

#pragma once

#include "IOptimizer.hpp"

namespace ORB_SLAM3
{

class GtsamOptimizer : public IOptimizer
{
public:
    void BundleAdjustment(const std::vector<KeyFrame*>& vpKF,
        const std::vector<MapPoint*>& vpMP, int nIterations = 5,
        bool* pbStopFlag = nullptr, unsigned long nLoopKF = 0,
        bool bRobust = true) override;

    void GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5,
        bool* pbStopFlag = nullptr, unsigned long nLoopKF = 0,
        bool bRobust = true) override;

    void FullInertialBA(Map* pMap, int its, bool bFixLocal = false,
        unsigned long nLoopKF = 0, bool* pbStopFlag = nullptr,
        bool bInit = false, float priorG = 1e2, float priorA = 1e6,
        Eigen::VectorXd* vSingVal = nullptr, bool* bHess = nullptr) override;

    void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap,
        int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges) override;

    void LocalBundleAdjustment(KeyFrame* pMainKF,
        std::vector<KeyFrame*> vpAdjustKF, std::vector<KeyFrame*> vpFixedKF,
        bool* pbStopFlag) override;

    int PoseOptimization(Frame* pFrame) override;
    int PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit = false) override;
    int PoseInertialOptimizationLastFrame(Frame* pFrame, bool bRecInit = false) override;

    void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
        const KeyFrameAndPose& NonCorrectedSim3, const KeyFrameAndPose& CorrectedSim3,
        const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
        const bool& bFixScale) override;

    void OptimizeEssentialGraph(KeyFrame* pCurKF,
        std::vector<KeyFrame*>& vpFixedKFs, std::vector<KeyFrame*>& vpFixedCorrectedKFs,
        std::vector<KeyFrame*>& vpNonFixedKFs,
        std::vector<MapPoint*>& vpNonCorrectedMPs) override;

    void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
        const KeyFrameAndPose& NonCorrectedSim3, const KeyFrameAndPose& CorrectedSim3,
        const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections) override;

    int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
        std::vector<MapPoint*>& vpMatches1, Sim3Type& g2oS12, float th2,
        bool bFixScale, Eigen::Matrix<double,7,7>& mAcumHessian,
        bool bAllPoints = false) override;

    void LocalInertialBA(KeyFrame* pKF, bool* pbStopFlag, Map* pMap,
        int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges,
        bool bLarge = false, bool bRecInit = false) override;

    void MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF,
        bool* pbStopFlag, Map* pMap, KeyFrameAndPose& corrPoses) override;

    void InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale,
        Eigen::Vector3d& bg, Eigen::Vector3d& ba, bool bMono,
        Eigen::MatrixXd& covInertial, bool bFixedVel = false,
        bool bGauss = false, float priorG = 1e2, float priorA = 1e6) override;

    void InertialOptimization(Map* pMap, Eigen::Vector3d& bg,
        Eigen::Vector3d& ba, float priorG = 1e2, float priorA = 1e6) override;

    void InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale) override;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

} // namespace ORB_SLAM3
