#pragma once

#include <vector>
#include <map>
#include <set>
#include <Eigen/Core>

namespace ORB_SLAM3 {

class Frame;
class KeyFrame;
class MapPoint;
class Map;
struct Sim3Type;
using KeyFrameAndPose = std::map<KeyFrame*, Sim3Type, std::less<KeyFrame*>,
    Eigen::aligned_allocator<std::pair<KeyFrame* const, Sim3Type>>>;

class IOptimizer {
public:
    virtual ~IOptimizer() = default;

    // Bundle Adjustment
    virtual void BundleAdjustment(const std::vector<KeyFrame*>& vpKF,
        const std::vector<MapPoint*>& vpMP, int nIterations = 5,
        bool* pbStopFlag = nullptr, unsigned long nLoopKF = 0, bool bRobust = true) = 0;

    virtual void GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5,
        bool* pbStopFlag = nullptr, unsigned long nLoopKF = 0, bool bRobust = true) = 0;

    virtual void FullInertialBA(Map* pMap, int its, bool bFixLocal = false,
        unsigned long nLoopKF = 0, bool* pbStopFlag = nullptr, bool bInit = false,
        float priorG = 1e2, float priorA = 1e6,
        Eigen::VectorXd* vSingVal = nullptr, bool* bHess = nullptr) = 0;

    virtual void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap,
        int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges) = 0;

    virtual void LocalBundleAdjustment(KeyFrame* pMainKF,
        std::vector<KeyFrame*> vpAdjustKF, std::vector<KeyFrame*> vpFixedKF,
        bool* pbStopFlag) = 0;

    // Pose Optimization
    virtual int PoseOptimization(Frame* pFrame) = 0;
    virtual int PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit = false) = 0;
    virtual int PoseInertialOptimizationLastFrame(Frame* pFrame, bool bRecInit = false) = 0;

    // Essential Graph / Sim3
    virtual void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
        const KeyFrameAndPose& NonCorrectedSim3, const KeyFrameAndPose& CorrectedSim3,
        const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
        const bool& bFixScale) = 0;

    virtual void OptimizeEssentialGraph(KeyFrame* pCurKF,
        std::vector<KeyFrame*>& vpFixedKFs, std::vector<KeyFrame*>& vpFixedCorrectedKFs,
        std::vector<KeyFrame*>& vpNonFixedKFs, std::vector<MapPoint*>& vpNonCorrectedMPs) = 0;

    virtual void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
        const KeyFrameAndPose& NonCorrectedSim3, const KeyFrameAndPose& CorrectedSim3,
        const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections) = 0;

    virtual int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
        std::vector<MapPoint*>& vpMatches1, Sim3Type& g2oS12, float th2,
        bool bFixScale, Eigen::Matrix<double,7,7>& mAcumHessian, bool bAllPoints = false) = 0;

    // Inertial
    virtual void LocalInertialBA(KeyFrame* pKF, bool* pbStopFlag, Map* pMap,
        int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges,
        bool bLarge = false, bool bRecInit = false) = 0;

    virtual void MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF,
        bool* pbStopFlag, Map* pMap, KeyFrameAndPose& corrPoses) = 0;

    virtual void InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale,
        Eigen::Vector3d& bg, Eigen::Vector3d& ba, bool bMono,
        Eigen::MatrixXd& covInertial, bool bFixedVel = false, bool bGauss = false,
        float priorG = 1e2, float priorA = 1e6) = 0;

    virtual void InertialOptimization(Map* pMap, Eigen::Vector3d& bg,
        Eigen::Vector3d& ba, float priorG = 1e2, float priorA = 1e6) = 0;

    virtual void InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale) = 0;
};

} // namespace ORB_SLAM3
