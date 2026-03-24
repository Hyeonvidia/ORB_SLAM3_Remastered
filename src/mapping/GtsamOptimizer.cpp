// =============================================================================
// ORB_SLAM3_Remastered -- GtsamOptimizer Implementation
// GTSAM backend for IOptimizer interface.
// PoseOptimization is fully implemented; all other methods throw until ported.
// =============================================================================
#include "GtsamOptimizer.hpp"
#include "Frame.hpp"
#include "KeyFrame.hpp"
#include "MapPoint.hpp"
#include "Map.hpp"
#include "GeometricCamera.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

#include <boost/optional.hpp>
#include <mutex>
#include <cmath>
#include <stdexcept>

namespace ORB_SLAM3
{

using gtsam::symbol_shorthand::X;  // Pose keys: X(i)

// ===========================================================================
// Custom GTSAM Factor: Monocular Reprojection (Pose-Only)
// Fixed 3D point, observed as (u, v), optimize only pose
// ===========================================================================
class MonoOnlyPoseFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    Eigen::Vector2d measured_;       // observed (u, v)
    Eigen::Vector3d Xw_;             // fixed world point
    GeometricCamera* pCamera_;       // camera model for projection

public:
    MonoOnlyPoseFactor(gtsam::Key poseKey,
                       const Eigen::Vector2d& measured,
                       const Eigen::Vector3d& Xw,
                       GeometricCamera* pCam,
                       const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey),
          measured_(measured), Xw_(Xw), pCamera_(pCam) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& Twc,
                                 boost::optional<gtsam::Matrix&> H = boost::none) const override {
        gtsam::Matrix36 Hpose;
        Eigen::Vector3d Pc = Twc.transformTo(Xw_, Hpose);

        if (Pc(2) <= 0.0) {
            // Behind camera -- return zero error to avoid distorting optimization
            if (H) *H = gtsam::Matrix::Zero(2, 6);
            return gtsam::Vector2::Zero();
        }

        // Project using camera model (takes Eigen::Vector3d, returns Eigen::Vector2d)
        Eigen::Vector2d proj = pCamera_->project(Pc);
        gtsam::Vector2 error = proj - measured_;

        if (H) {
            // projectJac takes Eigen::Vector3d, returns Eigen::Matrix<double,2,3>
            Eigen::Matrix<double, 2, 3> Jproj = pCamera_->projectJac(Pc);
            *H = Jproj * Hpose;
        }

        return error;
    }

    /// Compute squared error for outlier classification (without noise weighting)
    double chi2val(const gtsam::Pose3& Twc) const {
        gtsam::Vector e = evaluateError(Twc);
        return e.squaredNorm();
    }
};

// ===========================================================================
// Custom GTSAM Factor: Stereo Reprojection (Pose-Only)
// Fixed 3D point, observed as (uL, v, uR), optimize only pose
// ===========================================================================
class StereoOnlyPoseFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    Eigen::Vector3d measured_;  // (uL, v, uR)
    Eigen::Vector3d Xw_;
    double fx_, fy_, cx_, cy_, bf_;

public:
    StereoOnlyPoseFactor(gtsam::Key poseKey,
                          const Eigen::Vector3d& measured,
                          const Eigen::Vector3d& Xw,
                          double fx, double fy, double cx, double cy, double bf,
                          const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey),
          measured_(measured), Xw_(Xw), fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& Twc,
                                 boost::optional<gtsam::Matrix&> H = boost::none) const override {
        gtsam::Matrix36 Hpose;
        Eigen::Vector3d Pc = Twc.transformTo(Xw_, Hpose);

        if (Pc(2) <= 0.0) {
            if (H) *H = gtsam::Matrix::Zero(3, 6);
            return gtsam::Vector3::Zero();
        }

        double invZ = 1.0 / Pc(2);
        double u = fx_ * Pc(0) * invZ + cx_;
        double v = fy_ * Pc(1) * invZ + cy_;
        double ur = u - bf_ * invZ;

        gtsam::Vector3 error;
        error << u - measured_(0), v - measured_(1), ur - measured_(2);

        if (H) {
            Eigen::Matrix<double, 3, 3> Jproj;
            double invZ2 = invZ * invZ;
            Jproj << fx_ * invZ, 0, -fx_ * Pc(0) * invZ2,
                     0, fy_ * invZ, -fy_ * Pc(1) * invZ2,
                     fx_ * invZ, 0, -(fx_ * Pc(0) - bf_) * invZ2;
            *H = Jproj * Hpose;
        }

        return error;
    }
};

// ===========================================================================
// Helper: Create robust noise model with Huber kernel
// ===========================================================================
static gtsam::SharedNoiseModel makeHuberNoise2(double invSigma2, double delta) {
    auto gaussian = gtsam::noiseModel::Isotropic::Sigma(2, 1.0 / std::sqrt(invSigma2));
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(delta), gaussian);
}

static gtsam::SharedNoiseModel makeHuberNoise3(double invSigma2, double delta) {
    auto gaussian = gtsam::noiseModel::Isotropic::Sigma(3, 1.0 / std::sqrt(invSigma2));
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(delta), gaussian);
}

static gtsam::SharedNoiseModel makeGaussianNoise2(double invSigma2) {
    return gtsam::noiseModel::Isotropic::Sigma(2, 1.0 / std::sqrt(invSigma2));
}

static gtsam::SharedNoiseModel makeGaussianNoise3(double invSigma2) {
    return gtsam::noiseModel::Isotropic::Sigma(3, 1.0 / std::sqrt(invSigma2));
}

// ===========================================================================
// PoseOptimization -- pose-only optimization with fixed map points
// 4 rounds of LM optimization with outlier filtering
//
// Key convention: GTSAM uses Twc (world-to-camera), ORB-SLAM3 uses Tcw.
// We convert at the boundaries.
// ===========================================================================
int GtsamOptimizer::PoseOptimization(Frame* pFrame) {
    int nInitialCorrespondences = 0;
    const int N = pFrame->N;

    const float deltaMono = std::sqrt(5.991f);
    const float deltaStereo = std::sqrt(7.815f);

    // Collect observations
    struct MonoObs {
        int idx;
        Eigen::Vector2d obs;
        Eigen::Vector3d Xw;
        float invSigma2;
        GeometricCamera* pCam;
    };
    struct StereoObs {
        int idx;
        Eigen::Vector3d obs;
        Eigen::Vector3d Xw;
        float invSigma2;
    };

    std::vector<MonoObs> monoObs;
    std::vector<StereoObs> stereoObs;
    monoObs.reserve(N);
    stereoObs.reserve(N);

    {
        std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
        for (int i = 0; i < N; i++) {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (!pMP) continue;

            nInitialCorrespondences++;
            pFrame->mvbOutlier[i] = false;

            if (pFrame->mvuRight[i] < 0) {
                // Monocular observation
                const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                Eigen::Vector2d ob;
                ob << kpUn.pt.x, kpUn.pt.y;
                monoObs.push_back({i, ob, pMP->GetWorldPos().cast<double>(),
                                   pFrame->mvInvLevelSigma2[kpUn.octave], pFrame->mpCamera});
            } else {
                // Stereo observation
                const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                Eigen::Vector3d ob;
                ob << kpUn.pt.x, kpUn.pt.y, pFrame->mvuRight[i];
                stereoObs.push_back({i, ob, pMP->GetWorldPos().cast<double>(),
                                     pFrame->mvInvLevelSigma2[kpUn.octave]});
            }
        }
    }

    if (nInitialCorrespondences < 3) return 0;

    // 4 iterations of optimization with outlier filtering
    const float chi2Mono[4] = {5.991f, 5.991f, 5.991f, 5.991f};
    const float chi2Stereo[4] = {7.815f, 7.815f, 7.815f, 7.815f};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    for (int round = 0; round < 4; round++) {
        // Build factor graph
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;

        // Current pose estimate -- convert Tcw (ORB-SLAM3) to Twc (GTSAM)
        Sophus::SE3f Tcw_sophus = pFrame->GetPose();
        Eigen::Matrix4d Twc_matrix = Tcw_sophus.cast<double>().matrix().inverse();
        gtsam::Pose3 currentPose(Twc_matrix);
        initial.insert(X(0), currentPose);

        // Add monocular factors (excluding outliers after first round)
        for (auto& mo : monoObs) {
            if (pFrame->mvbOutlier[mo.idx] && round > 0) continue;

            gtsam::SharedNoiseModel noise;
            if (round < 2) {
                noise = makeHuberNoise2(mo.invSigma2, deltaMono);
            } else {
                noise = makeGaussianNoise2(mo.invSigma2);
            }

            graph.emplace_shared<MonoOnlyPoseFactor>(
                X(0), mo.obs, mo.Xw, mo.pCam, noise);
        }

        // Add stereo factors (excluding outliers after first round)
        for (auto& so : stereoObs) {
            if (pFrame->mvbOutlier[so.idx] && round > 0) continue;

            gtsam::SharedNoiseModel noise;
            if (round < 2) {
                noise = makeHuberNoise3(so.invSigma2, deltaStereo);
            } else {
                noise = makeGaussianNoise3(so.invSigma2);
            }

            graph.emplace_shared<StereoOnlyPoseFactor>(
                X(0), so.obs, so.Xw,
                pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy, pFrame->mbf,
                noise);
        }

        if (graph.size() < 3) break;

        // Optimize
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = its[round];
        params.setVerbosity("SILENT");
        params.absoluteErrorTol = 0;
        params.relativeErrorTol = 0;
        try {
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
            gtsam::Values result = optimizer.optimize();
            currentPose = result.at<gtsam::Pose3>(X(0));
            // Convert Twc (GTSAM) back to Tcw (ORB-SLAM3) via Sophus
            Sophus::SE3f Tcw_result = Sophus::SE3d(currentPose.inverse().matrix()).cast<float>();
            pFrame->SetPose(Tcw_result);
        } catch (const std::exception&) {
            // Optimization failed -- keep current pose
            break;
        }

        // Classify inliers/outliers
        nBad = 0;
        for (auto& mo : monoObs) {
            MonoOnlyPoseFactor f(X(0), mo.obs, mo.Xw, mo.pCam,
                                 gtsam::noiseModel::Unit::Create(2));
            double chi2 = f.chi2val(currentPose) * mo.invSigma2;
            if (chi2 > chi2Mono[round]) {
                pFrame->mvbOutlier[mo.idx] = true;
                nBad++;
            } else {
                pFrame->mvbOutlier[mo.idx] = false;
            }
        }

        for (auto& so : stereoObs) {
            // Compute stereo chi2 manually
            Eigen::Vector3d Pc = currentPose.transformTo(so.Xw);
            if (Pc(2) <= 0) {
                pFrame->mvbOutlier[so.idx] = true;
                nBad++;
                continue;
            }
            double invZ = 1.0 / Pc(2);
            double u  = pFrame->fx * Pc(0) * invZ + pFrame->cx;
            double v  = pFrame->fy * Pc(1) * invZ + pFrame->cy;
            double ur = u - pFrame->mbf * invZ;
            Eigen::Vector3d err;
            err << u - so.obs(0), v - so.obs(1), ur - so.obs(2);
            double chi2 = err.squaredNorm() * so.invSigma2;
            if (chi2 > chi2Stereo[round]) {
                pFrame->mvbOutlier[so.idx] = true;
                nBad++;
            } else {
                pFrame->mvbOutlier[so.idx] = false;
            }
        }
    }

    return nInitialCorrespondences - nBad;
}

// ===========================================================================
// Stub implementations -- throw until ported
// ===========================================================================

void GtsamOptimizer::BundleAdjustment(const std::vector<KeyFrame*>& /*vpKF*/,
    const std::vector<MapPoint*>& /*vpMP*/, int /*nIterations*/,
    bool* /*pbStopFlag*/, unsigned long /*nLoopKF*/, bool /*bRobust*/) {
    throw std::runtime_error("GtsamOptimizer::BundleAdjustment not yet implemented");
}

void GtsamOptimizer::GlobalBundleAdjustemnt(Map* /*pMap*/, int /*nIterations*/,
    bool* /*pbStopFlag*/, unsigned long /*nLoopKF*/, bool /*bRobust*/) {
    throw std::runtime_error("GtsamOptimizer::GlobalBundleAdjustemnt not yet implemented");
}

void GtsamOptimizer::FullInertialBA(Map* /*pMap*/, int /*its*/, bool /*bFixLocal*/,
    unsigned long /*nLoopKF*/, bool* /*pbStopFlag*/, bool /*bInit*/,
    float /*priorG*/, float /*priorA*/, Eigen::VectorXd* /*vSingVal*/, bool* /*bHess*/) {
    throw std::runtime_error("GtsamOptimizer::FullInertialBA not yet implemented");
}

void GtsamOptimizer::LocalBundleAdjustment(KeyFrame* /*pKF*/, bool* /*pbStopFlag*/,
    Map* /*pMap*/, int& /*num_fixedKF*/, int& /*num_OptKF*/,
    int& /*num_MPs*/, int& /*num_edges*/) {
    throw std::runtime_error("GtsamOptimizer::LocalBundleAdjustment not yet implemented");
}

void GtsamOptimizer::LocalBundleAdjustment(KeyFrame* /*pMainKF*/,
    std::vector<KeyFrame*> /*vpAdjustKF*/, std::vector<KeyFrame*> /*vpFixedKF*/,
    bool* /*pbStopFlag*/) {
    throw std::runtime_error("GtsamOptimizer::LocalBundleAdjustment not yet implemented");
}

int GtsamOptimizer::PoseInertialOptimizationLastKeyFrame(Frame* /*pFrame*/, bool /*bRecInit*/) {
    throw std::runtime_error("GtsamOptimizer::PoseInertialOptimizationLastKeyFrame not yet implemented");
}

int GtsamOptimizer::PoseInertialOptimizationLastFrame(Frame* /*pFrame*/, bool /*bRecInit*/) {
    throw std::runtime_error("GtsamOptimizer::PoseInertialOptimizationLastFrame not yet implemented");
}

void GtsamOptimizer::OptimizeEssentialGraph(Map* /*pMap*/, KeyFrame* /*pLoopKF*/,
    KeyFrame* /*pCurKF*/, const KeyFrameAndPose& /*NonCorrectedSim3*/,
    const KeyFrameAndPose& /*CorrectedSim3*/,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& /*LoopConnections*/,
    const bool& /*bFixScale*/) {
    throw std::runtime_error("GtsamOptimizer::OptimizeEssentialGraph not yet implemented");
}

void GtsamOptimizer::OptimizeEssentialGraph(KeyFrame* /*pCurKF*/,
    std::vector<KeyFrame*>& /*vpFixedKFs*/, std::vector<KeyFrame*>& /*vpFixedCorrectedKFs*/,
    std::vector<KeyFrame*>& /*vpNonFixedKFs*/, std::vector<MapPoint*>& /*vpNonCorrectedMPs*/) {
    throw std::runtime_error("GtsamOptimizer::OptimizeEssentialGraph not yet implemented");
}

void GtsamOptimizer::OptimizeEssentialGraph4DoF(Map* /*pMap*/, KeyFrame* /*pLoopKF*/,
    KeyFrame* /*pCurKF*/, const KeyFrameAndPose& /*NonCorrectedSim3*/,
    const KeyFrameAndPose& /*CorrectedSim3*/,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& /*LoopConnections*/) {
    throw std::runtime_error("GtsamOptimizer::OptimizeEssentialGraph4DoF not yet implemented");
}

int GtsamOptimizer::OptimizeSim3(KeyFrame* /*pKF1*/, KeyFrame* /*pKF2*/,
    std::vector<MapPoint*>& /*vpMatches1*/, Sim3Type& /*g2oS12*/, float /*th2*/,
    bool /*bFixScale*/, Eigen::Matrix<double,7,7>& /*mAcumHessian*/, bool /*bAllPoints*/) {
    throw std::runtime_error("GtsamOptimizer::OptimizeSim3 not yet implemented");
}

void GtsamOptimizer::LocalInertialBA(KeyFrame* /*pKF*/, bool* /*pbStopFlag*/,
    Map* /*pMap*/, int& /*num_fixedKF*/, int& /*num_OptKF*/,
    int& /*num_MPs*/, int& /*num_edges*/, bool /*bLarge*/, bool /*bRecInit*/) {
    throw std::runtime_error("GtsamOptimizer::LocalInertialBA not yet implemented");
}

void GtsamOptimizer::MergeInertialBA(KeyFrame* /*pCurrKF*/, KeyFrame* /*pMergeKF*/,
    bool* /*pbStopFlag*/, Map* /*pMap*/, KeyFrameAndPose& /*corrPoses*/) {
    throw std::runtime_error("GtsamOptimizer::MergeInertialBA not yet implemented");
}

void GtsamOptimizer::InertialOptimization(Map* /*pMap*/, Eigen::Matrix3d& /*Rwg*/,
    double& /*scale*/, Eigen::Vector3d& /*bg*/, Eigen::Vector3d& /*ba*/, bool /*bMono*/,
    Eigen::MatrixXd& /*covInertial*/, bool /*bFixedVel*/, bool /*bGauss*/,
    float /*priorG*/, float /*priorA*/) {
    throw std::runtime_error("GtsamOptimizer::InertialOptimization not yet implemented");
}

void GtsamOptimizer::InertialOptimization(Map* /*pMap*/, Eigen::Vector3d& /*bg*/,
    Eigen::Vector3d& /*ba*/, float /*priorG*/, float /*priorA*/) {
    throw std::runtime_error("GtsamOptimizer::InertialOptimization not yet implemented");
}

void GtsamOptimizer::InertialOptimization(Map* /*pMap*/, Eigen::Matrix3d& /*Rwg*/,
    double& /*scale*/) {
    throw std::runtime_error("GtsamOptimizer::InertialOptimization not yet implemented");
}

} // namespace ORB_SLAM3
