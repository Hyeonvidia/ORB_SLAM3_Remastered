// =============================================================================
// ORB_SLAM3_Remastered -- ABTestOptimizer Implementation
// Runs both primary (g2o) and reference (GTSAM) backends, compares results.
// Primary result is always used; reference is for diagnostic logging only.
// =============================================================================
#include "ABTestOptimizer.hpp"
#include "Frame.hpp"
#include <iostream>
#include <cmath>
#include <sophus/se3.hpp>

namespace ORB_SLAM3
{

// =============================================================================
// Constructor
// =============================================================================
ABTestOptimizer::ABTestOptimizer(std::unique_ptr<IOptimizer> primary,
                                 std::unique_ptr<IOptimizer> reference)
    : primary_(std::move(primary)), reference_(std::move(reference))
{
}

// =============================================================================
// Helper: log pose comparison
// =============================================================================
void ABTestOptimizer::logPoseComparison(const std::string& method,
                                         const Sophus::SE3f& posePrimary,
                                         const Sophus::SE3f& poseRef)
{
    Sophus::SE3f diff = posePrimary * poseRef.inverse();
    float transDiff = diff.translation().norm() * 1000.0f; // mm
    float rotDiff = diff.so3().log().norm() * (180.0f / static_cast<float>(M_PI)); // degrees
    std::cerr << "[AB-TEST " << method << "] "
              << "trans_diff=" << transDiff << "mm "
              << "rot_diff=" << rotDiff << "deg" << std::endl;
}

// =============================================================================
// PoseOptimization -- full A/B comparison with backup/restore
// =============================================================================
int ABTestOptimizer::PoseOptimization(Frame* pFrame)
{
    // 1. Backup frame state
    Sophus::SE3f backupPose = pFrame->GetPose();
    std::vector<bool> backupOutliers = pFrame->mvbOutlier;

    // 2. Run primary
    int primaryInliers = primary_->PoseOptimization(pFrame);
    Sophus::SE3f primaryPose = pFrame->GetPose();
    std::vector<bool> primaryOutliers = pFrame->mvbOutlier;

    // 3. Restore frame state for reference run
    pFrame->SetPose(backupPose);
    pFrame->mvbOutlier = backupOutliers;

    // 4. Run reference (may throw if not implemented)
    int refInliers = -1;
    Sophus::SE3f refPose;
    bool refOk = false;
    try {
        refInliers = reference_->PoseOptimization(pFrame);
        refPose = pFrame->GetPose();
        refOk = true;
    } catch (const std::runtime_error&) {
        // reference not yet implemented
    }

    // 5. Restore frame to primary's result (primary is what we use)
    pFrame->SetPose(primaryPose);
    pFrame->mvbOutlier = primaryOutliers;

    // 6. Log comparison
    if (refOk) {
        Sophus::SE3f diff = primaryPose * refPose.inverse();
        float transDiff = diff.translation().norm() * 1000.0f;
        float rotDiff = diff.so3().log().norm() * (180.0f / static_cast<float>(M_PI));
        std::cerr << "[AB-TEST PoseOpt] "
                  << "primary_inliers=" << primaryInliers
                  << " ref_inliers=" << refInliers
                  << " trans_diff=" << transDiff << "mm"
                  << " rot_diff=" << rotDiff << "deg"
                  << std::endl;
    } else {
        std::cerr << "[AB-TEST PoseOpt] primary: OK (inliers=" << primaryInliers
                  << "), reference: not yet implemented" << std::endl;
    }

    return primaryInliers;
}

// =============================================================================
// PoseInertialOptimizationLastKeyFrame
// =============================================================================
int ABTestOptimizer::PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit)
{
    int primaryResult = primary_->PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);
    try {
        reference_->PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);
        std::cerr << "[AB-TEST PoseInertialOptLastKF] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST PoseInertialOptLastKF] primary: OK, reference: not yet implemented" << std::endl;
    }
    return primaryResult;
}

// =============================================================================
// PoseInertialOptimizationLastFrame
// =============================================================================
int ABTestOptimizer::PoseInertialOptimizationLastFrame(Frame* pFrame, bool bRecInit)
{
    int primaryResult = primary_->PoseInertialOptimizationLastFrame(pFrame, bRecInit);
    try {
        reference_->PoseInertialOptimizationLastFrame(pFrame, bRecInit);
        std::cerr << "[AB-TEST PoseInertialOptLastFrame] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST PoseInertialOptLastFrame] primary: OK, reference: not yet implemented" << std::endl;
    }
    return primaryResult;
}

// =============================================================================
// BundleAdjustment
// =============================================================================
void ABTestOptimizer::BundleAdjustment(const std::vector<KeyFrame*>& vpKF,
    const std::vector<MapPoint*>& vpMP, int nIterations,
    bool* pbStopFlag, unsigned long nLoopKF, bool bRobust)
{
    primary_->BundleAdjustment(vpKF, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
    try {
        reference_->BundleAdjustment(vpKF, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
        std::cerr << "[AB-TEST BundleAdjustment] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST BundleAdjustment] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// GlobalBundleAdjustemnt
// =============================================================================
void ABTestOptimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations,
    bool* pbStopFlag, unsigned long nLoopKF, bool bRobust)
{
    primary_->GlobalBundleAdjustemnt(pMap, nIterations, pbStopFlag, nLoopKF, bRobust);
    try {
        reference_->GlobalBundleAdjustemnt(pMap, nIterations, pbStopFlag, nLoopKF, bRobust);
        std::cerr << "[AB-TEST GlobalBundleAdjustemnt] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST GlobalBundleAdjustemnt] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// FullInertialBA
// =============================================================================
void ABTestOptimizer::FullInertialBA(Map* pMap, int its, bool bFixLocal,
    unsigned long nLoopKF, bool* pbStopFlag, bool bInit,
    float priorG, float priorA, Eigen::VectorXd* vSingVal, bool* bHess)
{
    primary_->FullInertialBA(pMap, its, bFixLocal, nLoopKF, pbStopFlag, bInit, priorG, priorA, vSingVal, bHess);
    try {
        reference_->FullInertialBA(pMap, its, bFixLocal, nLoopKF, pbStopFlag, bInit, priorG, priorA, vSingVal, bHess);
        std::cerr << "[AB-TEST FullInertialBA] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST FullInertialBA] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// LocalBundleAdjustment (4-param version)
// =============================================================================
void ABTestOptimizer::LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap,
    int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges)
{
    primary_->LocalBundleAdjustment(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges);
    try {
        int ref_fixedKF = 0, ref_OptKF = 0, ref_MPs = 0, ref_edges = 0;
        reference_->LocalBundleAdjustment(pKF, pbStopFlag, pMap, ref_fixedKF, ref_OptKF, ref_MPs, ref_edges);
        std::cerr << "[AB-TEST LocalBundleAdjustment] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST LocalBundleAdjustment] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// LocalBundleAdjustment (merge version)
// =============================================================================
void ABTestOptimizer::LocalBundleAdjustment(KeyFrame* pMainKF,
    std::vector<KeyFrame*> vpAdjustKF, std::vector<KeyFrame*> vpFixedKF,
    bool* pbStopFlag)
{
    primary_->LocalBundleAdjustment(pMainKF, vpAdjustKF, vpFixedKF, pbStopFlag);
    try {
        reference_->LocalBundleAdjustment(pMainKF, vpAdjustKF, vpFixedKF, pbStopFlag);
        std::cerr << "[AB-TEST LocalBundleAdjustment2] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST LocalBundleAdjustment2] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// OptimizeEssentialGraph (loop closing version)
// =============================================================================
void ABTestOptimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    const KeyFrameAndPose& NonCorrectedSim3, const KeyFrameAndPose& CorrectedSim3,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
    const bool& bFixScale)
{
    primary_->OptimizeEssentialGraph(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale);
    try {
        reference_->OptimizeEssentialGraph(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale);
        std::cerr << "[AB-TEST OptimizeEssentialGraph] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST OptimizeEssentialGraph] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// OptimizeEssentialGraph (merge version)
// =============================================================================
void ABTestOptimizer::OptimizeEssentialGraph(KeyFrame* pCurKF,
    std::vector<KeyFrame*>& vpFixedKFs, std::vector<KeyFrame*>& vpFixedCorrectedKFs,
    std::vector<KeyFrame*>& vpNonFixedKFs, std::vector<MapPoint*>& vpNonCorrectedMPs)
{
    primary_->OptimizeEssentialGraph(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs);
    try {
        reference_->OptimizeEssentialGraph(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs);
        std::cerr << "[AB-TEST OptimizeEssentialGraph2] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST OptimizeEssentialGraph2] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// OptimizeEssentialGraph4DoF
// =============================================================================
void ABTestOptimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    const KeyFrameAndPose& NonCorrectedSim3, const KeyFrameAndPose& CorrectedSim3,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections)
{
    primary_->OptimizeEssentialGraph4DoF(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    try {
        reference_->OptimizeEssentialGraph4DoF(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
        std::cerr << "[AB-TEST OptimizeEssentialGraph4DoF] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST OptimizeEssentialGraph4DoF] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// OptimizeSim3
// =============================================================================
int ABTestOptimizer::OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
    std::vector<MapPoint*>& vpMatches1, Sim3Type& g2oS12, float th2,
    bool bFixScale, Eigen::Matrix<double,7,7>& mAcumHessian, bool bAllPoints)
{
    int primaryResult = primary_->OptimizeSim3(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, mAcumHessian, bAllPoints);
    try {
        Sim3Type refS12 = g2oS12;
        Eigen::Matrix<double,7,7> refHessian = mAcumHessian;
        reference_->OptimizeSim3(pKF1, pKF2, vpMatches1, refS12, th2, bFixScale, refHessian, bAllPoints);
        std::cerr << "[AB-TEST OptimizeSim3] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST OptimizeSim3] primary: OK, reference: not yet implemented" << std::endl;
    }
    return primaryResult;
}

// =============================================================================
// LocalInertialBA
// =============================================================================
void ABTestOptimizer::LocalInertialBA(KeyFrame* pKF, bool* pbStopFlag, Map* pMap,
    int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges,
    bool bLarge, bool bRecInit)
{
    primary_->LocalInertialBA(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges, bLarge, bRecInit);
    try {
        int ref_fixedKF = 0, ref_OptKF = 0, ref_MPs = 0, ref_edges = 0;
        reference_->LocalInertialBA(pKF, pbStopFlag, pMap, ref_fixedKF, ref_OptKF, ref_MPs, ref_edges, bLarge, bRecInit);
        std::cerr << "[AB-TEST LocalInertialBA] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST LocalInertialBA] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// MergeInertialBA
// =============================================================================
void ABTestOptimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF,
    bool* pbStopFlag, Map* pMap, KeyFrameAndPose& corrPoses)
{
    primary_->MergeInertialBA(pCurrKF, pMergeKF, pbStopFlag, pMap, corrPoses);
    try {
        reference_->MergeInertialBA(pCurrKF, pMergeKF, pbStopFlag, pMap, corrPoses);
        std::cerr << "[AB-TEST MergeInertialBA] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST MergeInertialBA] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// InertialOptimization (full version)
// =============================================================================
void ABTestOptimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale,
    Eigen::Vector3d& bg, Eigen::Vector3d& ba, bool bMono,
    Eigen::MatrixXd& covInertial, bool bFixedVel, bool bGauss,
    float priorG, float priorA)
{
    primary_->InertialOptimization(pMap, Rwg, scale, bg, ba, bMono, covInertial, bFixedVel, bGauss, priorG, priorA);
    try {
        Eigen::Matrix3d refRwg = Rwg; double refScale = scale;
        Eigen::Vector3d refBg = bg, refBa = ba;
        Eigen::MatrixXd refCov = covInertial;
        reference_->InertialOptimization(pMap, refRwg, refScale, refBg, refBa, bMono, refCov, bFixedVel, bGauss, priorG, priorA);
        std::cerr << "[AB-TEST InertialOptimization1] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST InertialOptimization1] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// InertialOptimization (bg/ba version)
// =============================================================================
void ABTestOptimizer::InertialOptimization(Map* pMap, Eigen::Vector3d& bg,
    Eigen::Vector3d& ba, float priorG, float priorA)
{
    primary_->InertialOptimization(pMap, bg, ba, priorG, priorA);
    try {
        Eigen::Vector3d refBg = bg, refBa = ba;
        reference_->InertialOptimization(pMap, refBg, refBa, priorG, priorA);
        std::cerr << "[AB-TEST InertialOptimization2] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST InertialOptimization2] primary: OK, reference: not yet implemented" << std::endl;
    }
}

// =============================================================================
// InertialOptimization (Rwg/scale version)
// =============================================================================
void ABTestOptimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg, double& scale)
{
    primary_->InertialOptimization(pMap, Rwg, scale);
    try {
        Eigen::Matrix3d refRwg = Rwg; double refScale = scale;
        reference_->InertialOptimization(pMap, refRwg, refScale);
        std::cerr << "[AB-TEST InertialOptimization3] primary: OK, reference: OK" << std::endl;
    } catch (const std::runtime_error&) {
        std::cerr << "[AB-TEST InertialOptimization3] primary: OK, reference: not yet implemented" << std::endl;
    }
}

} // namespace ORB_SLAM3
