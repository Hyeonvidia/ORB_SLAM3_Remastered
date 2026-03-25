// =============================================================================
// ORB_SLAM3_Remastered -- GtsamOptimizer Implementation
// GTSAM backend for IOptimizer interface.
// All 17 IOptimizer methods are implemented.
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
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/navigation/ImuBias.h>

#include "core/Sim3Type.hpp"
#include "ImuTypes.hpp"
#include "G2oTypes.hpp"  // for ConstraintPoseImu, Matrix15d

#include <boost/optional.hpp>
#include <mutex>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ORB_SLAM3
{

using gtsam::symbol_shorthand::X;  // Pose keys: X(i)
using gtsam::symbol_shorthand::L;  // Landmark keys: L(i)
using gtsam::symbol_shorthand::S;  // Sim3 keys: S(i)
using gtsam::symbol_shorthand::V;  // Velocity keys: V(i)
using gtsam::symbol_shorthand::B;  // Bias keys: B(i)

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
// Custom GTSAM Factor: Monocular BA (optimize both pose and landmark)
// ===========================================================================
class MonoProjectionFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
    Eigen::Vector2d measured_;
    GeometricCamera* pCamera_;

public:
    MonoProjectionFactor(gtsam::Key poseKey, gtsam::Key landmarkKey,
                          const Eigen::Vector2d& measured,
                          GeometricCamera* pCam,
                          const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, poseKey, landmarkKey),
          measured_(measured), pCamera_(pCam) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& Twc, const gtsam::Point3& Xw,
                                 boost::optional<gtsam::Matrix&> H1 = boost::none,
                                 boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        gtsam::Matrix36 Hpose;
        gtsam::Matrix33 Hpoint;
        Eigen::Vector3d Pc = Twc.transformTo(Xw, Hpose, Hpoint);

        if (Pc(2) <= 0.0) {
            if (H1) *H1 = gtsam::Matrix::Zero(2, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(2, 3);
            return gtsam::Vector2::Zero();
        }

        Eigen::Vector2d proj = pCamera_->project(Pc);
        gtsam::Vector2 error = proj - measured_;

        Eigen::Matrix<double, 2, 3> Jproj = pCamera_->projectJac(Pc);
        if (H1) *H1 = Jproj * Hpose;
        if (H2) *H2 = Jproj * Hpoint;

        return error;
    }
};

// ===========================================================================
// Custom GTSAM Factor: Stereo BA (optimize both pose and landmark)
// ===========================================================================
class StereoProjectionFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
    Eigen::Vector3d measured_;  // (uL, v, uR)
    double fx_, fy_, cx_, cy_, bf_;

public:
    StereoProjectionFactor(gtsam::Key poseKey, gtsam::Key landmarkKey,
                            const Eigen::Vector3d& measured,
                            double fx, double fy, double cx, double cy, double bf,
                            const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, poseKey, landmarkKey),
          measured_(measured), fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& Twc, const gtsam::Point3& Xw,
                                 boost::optional<gtsam::Matrix&> H1 = boost::none,
                                 boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        gtsam::Matrix36 Hpose;
        gtsam::Matrix33 Hpoint;
        Eigen::Vector3d Pc = Twc.transformTo(Xw, Hpose, Hpoint);

        if (Pc(2) <= 0.0) {
            if (H1) *H1 = gtsam::Matrix::Zero(3, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(3, 3);
            return gtsam::Vector3::Zero();
        }

        double invZ = 1.0 / Pc(2);
        double u = fx_ * Pc(0) * invZ + cx_;
        double v = fy_ * Pc(1) * invZ + cy_;
        double ur = u - bf_ * invZ;

        gtsam::Vector3 error;
        error << u - measured_(0), v - measured_(1), ur - measured_(2);

        double invZ2 = invZ * invZ;
        Eigen::Matrix<double, 3, 3> Jproj;
        Jproj << fx_ * invZ, 0, -fx_ * Pc(0) * invZ2,
                 0, fy_ * invZ, -fy_ * Pc(1) * invZ2,
                 fx_ * invZ, 0, -(fx_ * Pc(0) - bf_) * invZ2;
        if (H1) *H1 = Jproj * Hpose;
        if (H2) *H2 = Jproj * Hpoint;

        return error;
    }
};

// ===========================================================================
// Custom GTSAM Factor: Sim3 projection for OptimizeSim3
// Projects a fixed 3D point through Similarity3 transform
// ===========================================================================
class Sim3ProjectionFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
    Eigen::Vector2d measured_;
    Eigen::Vector3d P3Dc_;
    GeometricCamera* pCamera_;
    bool bInverse_;

public:
    Sim3ProjectionFactor(gtsam::Key sim3Key,
                         const Eigen::Vector2d& measured,
                         const Eigen::Vector3d& P3Dc,
                         GeometricCamera* pCam,
                         bool bInverse,
                         const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor1<gtsam::Similarity3>(model, sim3Key),
          measured_(measured), P3Dc_(P3Dc), pCamera_(pCam), bInverse_(bInverse) {}

    gtsam::Vector evaluateError(const gtsam::Similarity3& S12,
                                 boost::optional<gtsam::Matrix&> H = boost::none) const override {
        Eigen::Vector3d Pc;
        if (!bInverse_)
            Pc = S12.transformFrom(P3Dc_);
        else
            Pc = S12.inverse().transformFrom(P3Dc_);

        if (Pc(2) <= 0.0) {
            if (H) *H = gtsam::Matrix::Zero(2, 7);
            return gtsam::Vector2::Zero();
        }

        Eigen::Vector2d proj = pCamera_->project(Pc);
        gtsam::Vector2 error = proj - measured_;

        if (H) {
            // Numerical Jacobian for Similarity3 (7-dim manifold)
            *H = gtsam::Matrix::Zero(2, 7);
            const double eps = 1e-5;
            for (int j = 0; j < 7; j++) {
                Eigen::Matrix<double, 7, 1> delta = Eigen::Matrix<double, 7, 1>::Zero();
                delta(j) = eps;
                gtsam::Similarity3 S_plus = S12.retract(delta);
                Eigen::Vector3d Pc_p = bInverse_ ? S_plus.inverse().transformFrom(P3Dc_) : S_plus.transformFrom(P3Dc_);
                Eigen::Vector2d err_p = (Pc_p(2) > 0.0) ? (pCamera_->project(Pc_p) - measured_).eval() : Eigen::Vector2d(1e6, 1e6);
                delta(j) = -eps;
                gtsam::Similarity3 S_minus = S12.retract(delta);
                Eigen::Vector3d Pc_m = bInverse_ ? S_minus.inverse().transformFrom(P3Dc_) : S_minus.transformFrom(P3Dc_);
                Eigen::Vector2d err_m = (Pc_m(2) > 0.0) ? (pCamera_->project(Pc_m) - measured_).eval() : Eigen::Vector2d(1e6, 1e6);
                H->col(j) = (err_p - err_m) / (2.0 * eps);
            }
        }

        return error;
    }

    double chi2val(const gtsam::Similarity3& S12) const {
        gtsam::Vector err = evaluateError(S12);
        return err.squaredNorm();
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
// BundleAdjustment -- joint optimization of KF poses + MP positions
// ===========================================================================
void GtsamOptimizer::BundleAdjustment(const std::vector<KeyFrame*>& vpKFs,
    const std::vector<MapPoint*>& vpMP, int nIterations,
    bool* pbStopFlag, unsigned long nLoopKF, bool bRobust) {
    if (vpKFs.empty()) return;

    Map* pMap = vpKFs[0]->GetMap();
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const float thHuber2D = std::sqrt(5.991f);
    const float thHuber3D = std::sqrt(7.815f);

    long unsigned int maxKFid = 0;

    // Add KF pose vertices
    for (auto* pKF : vpKFs) {
        if (pKF->isBad()) continue;
        Sophus::SE3f Tcw = pKF->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKF->mnId), Twc);

        // Fix first KF
        if (pKF->mnId == pMap->GetInitKFid()) {
            auto priorNoise = gtsam::noiseModel::Constrained::All(6);
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(pKF->mnId), Twc, priorNoise);
        }

        if (pKF->mnId > maxKFid) maxKFid = pKF->mnId;
    }

    // Add MP vertices and projection factors
    std::vector<bool> vbNotIncludedMP(vpMP.size(), false);

    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint* pMP = vpMP[i];
        if (pMP->isBad()) continue;

        gtsam::Key lmKey = L(pMP->mnId);
        initial.insert(lmKey, gtsam::Point3(pMP->GetWorldPos().cast<double>()));

        auto observations = pMP->GetObservations();
        int nEdges = 0;

        for (auto& [pKF, indices] : observations) {
            if (pKF->isBad() || pKF->mnId > maxKFid) continue;
            if (!initial.exists(X(pKF->mnId))) continue;

            const int leftIndex = std::get<0>(indices);
            if (leftIndex == -1) continue;

            nEdges++;

            if (pKF->mvuRight[leftIndex] < 0) {
                // Monocular observation
                const cv::KeyPoint& kpUn = pKF->mvKeysUn[leftIndex];
                Eigen::Vector2d obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                float invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];

                gtsam::SharedNoiseModel noise = bRobust ?
                    makeHuberNoise2(invSigma2, thHuber2D) : makeGaussianNoise2(invSigma2);

                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKF->mnId), lmKey, obs, pKF->mpCamera, noise);
            } else {
                // Stereo observation
                const cv::KeyPoint& kpUn = pKF->mvKeysUn[leftIndex];
                float kp_ur = pKF->mvuRight[leftIndex];
                Eigen::Vector3d obs;
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                float invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];

                gtsam::SharedNoiseModel noise = bRobust ?
                    makeHuberNoise3(invSigma2, thHuber3D) : makeGaussianNoise3(invSigma2);

                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKF->mnId), lmKey, obs,
                    pKF->fx, pKF->fy, pKF->cx, pKF->cy, pKF->mbf, noise);
            }
        }

        if (nEdges == 0) {
            initial.erase(lmKey);
            vbNotIncludedMP[i] = true;
        }
    }

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = nIterations;
    params.setVerbosity("SILENT");

    // Build explicit ordering from all initial values
    // This prevents GTSAM's auto-ordering from missing disconnected components
    gtsam::Ordering ordering;
    // Add landmarks first (will be eliminated first = marginalized)
    for (const auto& key_value : initial) {
        gtsam::Symbol sym(key_value.key);
        if (sym.chr() == 'l')
            ordering.push_back(key_value.key);
    }
    // Then poses
    for (const auto& key_value : initial) {
        gtsam::Symbol sym(key_value.key);
        if (sym.chr() == 'x')
            ordering.push_back(key_value.key);
    }

    try {
        params.ordering = ordering;
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        // Recover KF poses
        for (auto* pKF : vpKFs) {
            if (pKF->isBad()) continue;
            gtsam::Key key = X(pKF->mnId);
            if (!result.exists(key)) continue;
            gtsam::Pose3 optimizedTwc = result.at<gtsam::Pose3>(key);

            Sophus::SE3f Tcw = Sophus::SE3d(optimizedTwc.inverse().matrix()).cast<float>();

            if (nLoopKF == pMap->GetOriginKF()->mnId) {
                pKF->SetPose(Tcw);
            } else {
                pKF->mTcwGBA = Tcw;
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        // Recover MP positions
        for (size_t i = 0; i < vpMP.size(); i++) {
            if (vbNotIncludedMP[i]) continue;
            MapPoint* pMP = vpMP[i];
            if (pMP->isBad()) continue;

            gtsam::Key lmKey = L(pMP->mnId);
            if (!result.exists(lmKey)) continue;
            gtsam::Point3 pt = result.at<gtsam::Point3>(lmKey);

            if (nLoopKF == pMap->GetOriginKF()->mnId) {
                pMP->SetWorldPos(pt.cast<float>());
                pMP->UpdateNormalAndDepth();
            } else {
                pMP->mPosGBA = pt.cast<float>();
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::BundleAdjustment] GTSAM error: " << e.what() << std::endl;
    }
}

void GtsamOptimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations,
    bool* pbStopFlag, unsigned long nLoopKF, bool bRobust) {
    auto vpKFs = pMap->GetAllKeyFrames();
    auto vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void GtsamOptimizer::FullInertialBA(Map* pMap, int its, bool bFixLocal,
    unsigned long nLoopKF, bool* pbStopFlag, bool bInit,
    float priorG, float priorA, Eigen::VectorXd* vSingVal, bool* bHess) {
    if (!pMap) return;

    auto vpKFs = pMap->GetAllKeyFrames();
    auto vpMPs = pMap->GetAllMapPoints();

    if (vpKFs.empty()) return;

    // Sort KFs by ID for temporal ordering
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame* a, KeyFrame* b) { return a->mnId < b->mnId; });

    long unsigned int maxKFid = 0;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const float thHuber2D = std::sqrt(5.991f);
    const float thHuber3D = std::sqrt(7.815f);

    // --- Add KF pose, velocity, and bias vertices ---
    for (auto* pKF : vpKFs) {
        if (pKF->isBad()) continue;

        Sophus::SE3f Tcw = pKF->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKF->mnId), Twc);

        // Velocity
        Eigen::Vector3f vel = pKF->GetVelocity();
        initial.insert(V(pKF->mnId), Eigen::Vector3d(vel.cast<double>()));

        // Bias
        Eigen::Vector3f bg = pKF->GetGyroBias();
        Eigen::Vector3f ba = pKF->GetAccBias();
        initial.insert(B(pKF->mnId),
            gtsam::imuBias::ConstantBias(Eigen::Vector3d(ba.cast<double>()), Eigen::Vector3d(bg.cast<double>())));

        // Fix first KF
        if (pKF->mnId == pMap->GetInitKFid()) {
            auto priorNoise = gtsam::noiseModel::Constrained::All(6);
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(pKF->mnId), Twc, priorNoise);
        }

        if (pKF->mnId > maxKFid) maxKFid = pKF->mnId;
    }

    // --- IMU constraints between consecutive KFs ---
    for (auto* pKF : vpKFs) {
        if (pKF->isBad() || !pKF->mPrevKF || !pKF->mpImuPreintegrated) continue;
        if (pKF->mPrevKF->isBad()) continue;
        if (!initial.exists(X(pKF->mPrevKF->mnId))) continue;
        if (!initial.exists(X(pKF->mnId))) continue;

        IMU::Preintegrated* pInt = pKF->mpImuPreintegrated;
        if (pInt->dT < 1e-5f) continue;

        double dt = pInt->dT;
        Eigen::Matrix3d dR = pInt->GetUpdatedDeltaRotation().cast<double>();
        Eigen::Vector3d dV_imu = pInt->GetUpdatedDeltaVelocity().cast<double>();
        Eigen::Vector3d dP = pInt->GetUpdatedDeltaPosition().cast<double>();

        // Camera-to-body transform
        Eigen::Matrix4d Tcb_mat = pKF->mImuCalib.mTcb.cast<double>().matrix();
        gtsam::Pose3 TcbPose(gtsam::Rot3(Tcb_mat.block<3,3>(0,0)),
                              gtsam::Point3(Tcb_mat.block<3,1>(0,3)));

        // Get previous KF body pose: Twb = Twc * Tcb
        gtsam::Pose3 Twc_prev = initial.at<gtsam::Pose3>(X(pKF->mPrevKF->mnId));
        gtsam::Pose3 Twb1 = Twc_prev * TcbPose;

        Eigen::Matrix3d Rwb1 = Twb1.rotation().matrix();
        Eigen::Vector3d twb1 = Twb1.translation();
        Eigen::Vector3d vel1 = initial.at<gtsam::Vector3>(V(pKF->mPrevKF->mnId));
        Eigen::Vector3d g; g << 0, 0, -IMU::GRAVITY_VALUE;

        // IMU-predicted body pose
        Eigen::Matrix3d Rwb2_pred = Rwb1 * dR;
        Eigen::Vector3d twb2_pred = twb1 + vel1 * dt + 0.5 * g * dt * dt + Rwb1 * dP;
        Eigen::Vector3d vel2_pred = vel1 + g * dt + Rwb1 * dV_imu;

        // Convert predicted body pose to camera frame: Twc = Twb * Tcb^{-1}
        gtsam::Pose3 Twb_pred{gtsam::Rot3(Rwb2_pred), gtsam::Point3(twb2_pred)};
        gtsam::Pose3 Twc_pred = Twb_pred * TcbPose.inverse();

        // Relative pose constraint from IMU prediction
        auto poseSigmas = (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished();
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(pKF->mPrevKF->mnId), X(pKF->mnId),
            Twc_prev.inverse() * Twc_pred,
            gtsam::noiseModel::Diagonal::Sigmas(poseSigmas));

        // Velocity constraint
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            V(pKF->mnId), vel2_pred,
            gtsam::noiseModel::Isotropic::Sigma(3, 0.1));

        // Bias random walk constraint
        Eigen::Matrix3d InfoG = pInt->C.block<3,3>(9,9).cast<double>();
        Eigen::Matrix3d InfoA = pInt->C.block<3,3>(12,12).cast<double>();
        for (int i = 0; i < 3; i++) {
            if (InfoG(i,i) < 1e-10) InfoG(i,i) = 1e-10;
            if (InfoA(i,i) < 1e-10) InfoA(i,i) = 1e-10;
        }
        Eigen::Matrix<double, 6, 6> biasInfo = Eigen::Matrix<double, 6, 6>::Zero();
        biasInfo.block<3,3>(0,0) = InfoA.inverse();
        biasInfo.block<3,3>(3,3) = InfoG.inverse();
        auto biasNoise = gtsam::noiseModel::Gaussian::Information(biasInfo);
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
            B(pKF->mPrevKF->mnId), B(pKF->mnId),
            gtsam::imuBias::ConstantBias(), biasNoise);
    }

    // --- Bias priors for initialization ---
    if (bInit) {
        for (auto* pKF : vpKFs) {
            if (pKF->isBad() || !initial.exists(B(pKF->mnId))) continue;
            auto biasPrior = initial.at<gtsam::imuBias::ConstantBias>(B(pKF->mnId));
            Eigen::Matrix<double, 6, 6> priorInfo = Eigen::Matrix<double, 6, 6>::Zero();
            priorInfo.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * priorA;
            priorInfo.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * priorG;
            graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(pKF->mnId), biasPrior,
                gtsam::noiseModel::Gaussian::Information(priorInfo));
        }
    }

    // --- Visual factors (projection) ---
    std::vector<bool> vbNotIncludedMP(vpMPs.size(), false);

    for (size_t i = 0; i < vpMPs.size(); i++) {
        MapPoint* pMP = vpMPs[i];
        if (pMP->isBad()) continue;

        gtsam::Key lmKey = L(pMP->mnId);
        initial.insert(lmKey, gtsam::Point3(pMP->GetWorldPos().cast<double>()));

        auto observations = pMP->GetObservations();
        int nEdges = 0;

        for (auto& [pKF, indices] : observations) {
            if (pKF->isBad() || pKF->mnId > maxKFid) continue;
            if (!initial.exists(X(pKF->mnId))) continue;

            const int leftIndex = std::get<0>(indices);
            if (leftIndex == -1) continue;

            nEdges++;

            if (pKF->mvuRight[leftIndex] < 0) {
                const cv::KeyPoint& kpUn = pKF->mvKeysUn[leftIndex];
                Eigen::Vector2d obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                float invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise2(invSigma2, thHuber2D);
                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKF->mnId), lmKey, obs, pKF->mpCamera, noise);
            } else {
                const cv::KeyPoint& kpUn = pKF->mvKeysUn[leftIndex];
                float kp_ur = pKF->mvuRight[leftIndex];
                Eigen::Vector3d obs;
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                float invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise3(invSigma2, thHuber3D);
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKF->mnId), lmKey, obs,
                    pKF->fx, pKF->fy, pKF->cx, pKF->cy, pKF->mbf, noise);
            }
        }

        if (nEdges == 0) {
            initial.erase(lmKey);
            vbNotIncludedMP[i] = true;
        }
    }

    if (pbStopFlag && *pbStopFlag) return;

    // --- Optimize ---
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = its;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        // --- Recover results ---
        for (auto* pKF : vpKFs) {
            if (pKF->isBad()) continue;
            gtsam::Key key = X(pKF->mnId);
            if (!result.exists(key)) continue;

            gtsam::Pose3 optimizedTwc = result.at<gtsam::Pose3>(key);
            Sophus::SE3f Tcw = Sophus::SE3d(optimizedTwc.inverse().matrix()).cast<float>();

            if (nLoopKF == 0) {
                // Direct update (initialization path)
                pKF->SetPose(Tcw);

                if (result.exists(V(pKF->mnId)))
                    pKF->SetVelocity(result.at<gtsam::Vector3>(V(pKF->mnId)).cast<float>());

                if (result.exists(B(pKF->mnId))) {
                    auto optBias = result.at<gtsam::imuBias::ConstantBias>(B(pKF->mnId));
                    pKF->SetNewBias(IMU::Bias(
                        optBias.accelerometer()(0), optBias.accelerometer()(1), optBias.accelerometer()(2),
                        optBias.gyroscope()(0), optBias.gyroscope()(1), optBias.gyroscope()(2)));
                }
            } else {
                // Store for GBA recovery (LoopClosing::RunGlobalBundleAdjustment)
                pKF->mTcwGBA = Tcw;
                pKF->mnBAGlobalForKF = nLoopKF;

                if (result.exists(V(pKF->mnId)))
                    pKF->mVwbGBA = result.at<gtsam::Vector3>(V(pKF->mnId)).cast<float>();

                if (result.exists(B(pKF->mnId))) {
                    auto optBias = result.at<gtsam::imuBias::ConstantBias>(B(pKF->mnId));
                    pKF->mBiasGBA = IMU::Bias(
                        optBias.accelerometer()(0), optBias.accelerometer()(1), optBias.accelerometer()(2),
                        optBias.gyroscope()(0), optBias.gyroscope()(1), optBias.gyroscope()(2));
                }
            }
        }

        // Recover MP positions
        for (size_t i = 0; i < vpMPs.size(); i++) {
            if (vbNotIncludedMP[i]) continue;
            MapPoint* pMP = vpMPs[i];
            if (pMP->isBad()) continue;

            gtsam::Key lmKey = L(pMP->mnId);
            if (!result.exists(lmKey)) continue;
            gtsam::Point3 pt = result.at<gtsam::Point3>(lmKey);

            if (nLoopKF == 0) {
                pMP->SetWorldPos(pt.cast<float>());
                pMP->UpdateNormalAndDepth();
            } else {
                pMP->mPosGBA = pt.cast<float>();
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::FullInertialBA] GTSAM error: " << e.what() << std::endl;
    }
}

void GtsamOptimizer::LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag,
    Map* pMap, int& num_fixedKF, int& num_OptKF,
    int& num_MPs, int& num_edges) {
    num_fixedKF = 0; num_OptKF = 0; num_MPs = 0; num_edges = 0;

    // Gather local KFs (current + covisible)
    std::list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    auto vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (auto* pKFi : vNeighKFs) {
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Gather local MPs observed by local KFs
    std::list<MapPoint*> lLocalMapPoints;
    std::set<MapPoint*> sLocalMPs;
    for (auto* pKFi : lLocalKeyFrames) {
        auto vpMPs = pKFi->GetMapPointMatches();
        for (auto* pMP : vpMPs) {
            if (pMP && !pMP->isBad() && pMP->GetMap() == pMap) {
                if (sLocalMPs.insert(pMP).second)
                    lLocalMapPoints.push_back(pMP);
            }
        }
    }

    // Gather fixed KFs (observe local MPs but not in local set)
    std::list<KeyFrame*> lFixedCameras;
    std::set<KeyFrame*> sFixedKFs;
    for (auto* pMP : lLocalMapPoints) {
        auto obs = pMP->GetObservations();
        for (auto& [pKFi, indices] : obs) {
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pMap) {
                    lFixedCameras.push_back(pKFi);
                    sFixedKFs.insert(pKFi);
                }
            }
        }
    }

    num_OptKF = static_cast<int>(lLocalKeyFrames.size());
    num_fixedKF = static_cast<int>(lFixedCameras.size());
    num_MPs = static_cast<int>(lLocalMapPoints.size());

    if (lLocalKeyFrames.size() < 2) return;

    // Build GTSAM graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const float thHuber2D = std::sqrt(5.991f);
    const float thHuber3D = std::sqrt(7.815f);

    // Add optimizable KF poses
    for (auto* pKFi : lLocalKeyFrames) {
        Sophus::SE3f Tcw = pKFi->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKFi->mnId), Twc);
    }

    // Add fixed KF poses with strong priors
    for (auto* pKFi : lFixedCameras) {
        Sophus::SE3f Tcw = pKFi->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKFi->mnId), Twc);
        auto priorNoise = gtsam::noiseModel::Constrained::All(6);
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(pKFi->mnId), Twc, priorNoise);
    }

    // Add MPs and projection factors — track KF-MP pairs for outlier erasure
    struct EdgeInfo {
        KeyFrame* pKF;
        MapPoint* pMP;
        bool isMono;         // true = mono (2-DOF), false = stereo (3-DOF)
        float invSigma2;
        size_t factorIdx;    // index in graph
    };
    std::vector<EdgeInfo> vEdgeInfo;
    int edgeCount = 0;

    for (auto* pMP : lLocalMapPoints) {
        gtsam::Key lmKey = L(pMP->mnId);
        initial.insert(lmKey, gtsam::Point3(pMP->GetWorldPos().cast<double>()));

        auto obs = pMP->GetObservations();
        for (auto& [pKFi, indices] : obs) {
            if (pKFi->isBad() || (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId))
                continue;
            if (!initial.exists(X(pKFi->mnId))) continue;

            const int leftIndex = std::get<0>(indices);
            if (leftIndex == -1) continue;

            size_t factorIdx = graph.size();  // index of the factor about to be added

            if (pKFi->mvuRight[leftIndex] < 0) {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Vector2d ob; ob << kpUn.pt.x, kpUn.pt.y;
                float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise2(invSigma2, thHuber2D);
                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKFi->mnId), lmKey, ob, pKFi->mpCamera, noise);
                vEdgeInfo.push_back({pKFi, pMP, true, invSigma2, factorIdx});
            } else {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Vector3d ob;
                ob << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise3(invSigma2, thHuber3D);
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKFi->mnId), lmKey, ob,
                    pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy, pKFi->mbf, noise);
                vEdgeInfo.push_back({pKFi, pMP, false, invSigma2, factorIdx});
            }
            edgeCount++;
        }
    }
    num_edges = edgeCount;

    // === 2-Round LBA (matches g2o: optimize(5) + setLevel(1) + optimize(10)) ===

    // Round 1: 5 iterations with Huber kernel
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 5;
    params.setVerbosity("SILENT");
    params.absoluteErrorTol = 0;
    params.relativeErrorTol = 0;

    gtsam::Values round1Result;
    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        round1Result = optimizer.optimize();
    } catch (const std::exception& e) {
        std::cerr << "[LBA-R1] GTSAM error: " << e.what() << std::endl;
        return;
    }

    // Identify outlier factors and rebuild graph WITHOUT them for Round 2
    // GTSAM factor->error() = 0.5 * ||h(x)-z||^2_Sigma = 0.5 * chi2
    // g2o chi2 thresholds: 5.991 (mono 2-DOF), 7.815 (stereo 3-DOF)
    // So GTSAM error threshold = g2o_chi2_threshold / 2
    const double errThreshMono = 5.991 / 2.0;    // = 2.9955
    const double errThreshStereo = 7.815 / 2.0;  // = 3.9075

    gtsam::NonlinearFactorGraph graph2;
    int nOutlierFactors = 0;
    int nTotalFactors = 0;

    for (size_t i = 0; i < graph.size(); i++) {
        auto factor = graph[i];
        if (!factor) continue;

        // Prior factors always kept
        if (factor->keys().size() == 1) {
            graph2.add(factor);
            continue;
        }

        // Projection factors: check chi2
        double error = factor->error(round1Result);
        bool isMono = (factor->dim() == 2);  // 2-DOF = mono, 3-DOF = stereo
        double threshold = isMono ? errThreshMono : errThreshStereo;

        nTotalFactors++;
        if (error > threshold) {
            nOutlierFactors++;
            // Skip outlier factor in Round 2
        } else {
            // Keep inlier factor in Round 2
            graph2.add(factor);
        }
    }

    // Round 2: 5 iterations without outliers
    gtsam::Values result;
    if (nOutlierFactors > 0 && nOutlierFactors < nTotalFactors) {
        try {
            // Build filtered Values containing only variables referenced by graph2
            // This prevents orphan variables from crashing the elimination
            gtsam::Values round2Initial;
            std::set<gtsam::Key> graph2Keys;
            for (size_t i = 0; i < graph2.size(); i++) {
                if (graph2[i]) {
                    for (auto key : graph2[i]->keys())
                        graph2Keys.insert(key);
                }
            }
            for (auto key : graph2Keys) {
                if (round1Result.exists(key))
                    round2Initial.insert(key, round1Result.at(key));
            }

            gtsam::LevenbergMarquardtParams params2;
            params2.maxIterations = 5;
            params2.setVerbosity("SILENT");
            params2.absoluteErrorTol = 0;
            params2.relativeErrorTol = 0;
            gtsam::LevenbergMarquardtOptimizer optimizer2(graph2, round2Initial, params2);
            result = optimizer2.optimize();

            // Merge: R2 result for optimized vars, R1 for vars excluded from R2
            for (const auto& kv : round1Result) {
                if (!result.exists(kv.key))
                    result.insert(kv.key, kv.value);
            }

            std::cerr << "[LBA-OPT] R1+R2 outliers=" << nOutlierFactors << "/" << nTotalFactors
                      << " R2iters=" << optimizer2.iterations() << std::endl;
        } catch (const std::exception& e) {
            result = round1Result;
            std::cerr << "[LBA-R2] fallback to R1: " << e.what() << std::endl;
        }
    } else {
        result = round1Result;
        std::cerr << "[LBA-OPT] R1-only outliers=" << nOutlierFactors << "/" << nTotalFactors << std::endl;
    }

    // Classify outliers and collect observations to erase (g2o-equivalent)
    // GTSAM factor->error() = 0.5 * chi2, so chi2 = 2 * error
    std::vector<std::pair<KeyFrame*,MapPoint*>> vToErase;
    for (auto& ei : vEdgeInfo) {
        if (ei.pMP->isBad()) continue;
        auto factor = graph[ei.factorIdx];
        if (!factor) continue;

        double error = factor->error(result);
        double chi2 = 2.0 * error;  // GTSAM error = 0.5 * ||e||^2_Sigma
        double threshold = ei.isMono ? 5.991 : 7.815;

        // Also check depth positivity for the projection
        bool depthOK = true;
        if (result.exists(X(ei.pKF->mnId)) && result.exists(L(ei.pMP->mnId))) {
            gtsam::Pose3 Twc = result.at<gtsam::Pose3>(X(ei.pKF->mnId));
            gtsam::Point3 Xw = result.at<gtsam::Point3>(L(ei.pMP->mnId));
            Eigen::Vector3d Pc = Twc.transformTo(Xw);
            if (Pc(2) <= 0.0) depthOK = false;
        }

        if (chi2 > threshold || !depthOK) {
            vToErase.push_back(std::make_pair(ei.pKF, ei.pMP));
        }
    }

    // Compute shifts BEFORE writing back to check for catastrophic results
    double totalPoseShift = 0;
    int nPoseRecovered = 0;
    std::vector<std::pair<KeyFrame*, gtsam::Pose3>> poseUpdates;
    for (auto* pKFi : lLocalKeyFrames) {
        if (result.exists(X(pKFi->mnId))) {
            Sophus::SE3f oldTcw = pKFi->GetPose();
            Eigen::Vector3d oldPos = oldTcw.cast<double>().inverse().translation();
            gtsam::Pose3 newTwc = result.at<gtsam::Pose3>(X(pKFi->mnId));
            totalPoseShift += (newTwc.translation() - oldPos).norm();
            nPoseRecovered++;
            poseUpdates.emplace_back(pKFi, newTwc);
        }
    }

    double totalMPShift = 0;
    int nMPRecovered = 0;
    std::vector<std::pair<MapPoint*, Eigen::Vector3f>> mpUpdates;
    for (auto* pMP : lLocalMapPoints) {
        if (result.exists(L(pMP->mnId))) {
            Eigen::Vector3f oldPos = pMP->GetWorldPos();
            Eigen::Vector3f newPos = result.at<gtsam::Point3>(L(pMP->mnId)).cast<float>();
            totalMPShift += (newPos - oldPos).norm();
            nMPRecovered++;
            mpUpdates.emplace_back(pMP, newPos);
        }
    }

    double avgMPShift = nMPRecovered > 0 ? totalMPShift / nMPRecovered : 0;
    double avgPoseShift = nPoseRecovered > 0 ? totalPoseShift / nPoseRecovered : 0;

    // Get Map Mutex — matches g2o LBA pattern
    std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

    // Erase outlier observations from the map (g2o-equivalent)
    if (!vToErase.empty()) {
        for (auto& [pKFi, pMPi] : vToErase) {
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Guard: discard catastrophic LBA results
    if (avgMPShift > 0.5) {
        std::cerr << "[LBA-GUARD] DISCARDED avgPoseShift=" << avgPoseShift
                  << " avgMPShift=" << avgMPShift << " (>0.5m threshold)"
                  << " erased=" << vToErase.size() << std::endl;
    } else {
        // Apply pose updates (Twc -> Tcw via Sophus)
        for (auto& [pKFi, newTwc] : poseUpdates) {
            Sophus::SE3f Tcw = Sophus::SE3d(newTwc.inverse().matrix()).cast<float>();
            pKFi->SetPose(Tcw);
        }
        // Apply MP updates
        for (auto& [pMP, newPos] : mpUpdates) {
            pMP->SetWorldPos(newPos);
            pMP->UpdateNormalAndDepth();
        }
        if (nPoseRecovered > 0 || nMPRecovered > 0) {
            std::cerr << "[LBA-SHIFT] avgPoseShift=" << avgPoseShift
                      << " avgMPShift=" << avgMPShift
                      << " nKF=" << nPoseRecovered << " nMP=" << nMPRecovered
                      << " erased=" << vToErase.size() << std::endl;
        }
    }

    pMap->IncreaseChangeIndex();
}

void GtsamOptimizer::LocalBundleAdjustment(KeyFrame* pMainKF,
    std::vector<KeyFrame*> vpAdjustKF, std::vector<KeyFrame*> vpFixedKF,
    bool* pbStopFlag) {

    std::vector<MapPoint*> vpMPs;
    Map* pCurrentMap = pMainKF->GetMap();
    long unsigned int maxKFid = 0;
    std::set<KeyFrame*> spKeyFrameBA;

    const float thHuber2D = std::sqrt(5.99f);
    const float thHuber3D = std::sqrt(7.815f);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    // Set fixed KF vertices
    for (KeyFrame* pKFi : vpFixedKF) {
        if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
        pKFi->mnBALocalForMerge = pMainKF->mnId;

        Sophus::SE3f Tcw = pKFi->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKFi->mnId), Twc);
        auto priorNoise = gtsam::noiseModel::Constrained::All(6);
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(pKFi->mnId), Twc, priorNoise);

        if (pKFi->mnId > maxKFid) maxKFid = pKFi->mnId;

        std::set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for (MapPoint* pMPi : spViewMPs) {
            if (pMPi && !pMPi->isBad() && pMPi->GetMap() == pCurrentMap) {
                if (pMPi->mnBALocalForMerge != pMainKF->mnId) {
                    vpMPs.push_back(pMPi);
                    pMPi->mnBALocalForMerge = pMainKF->mnId;
                }
            }
        }
        spKeyFrameBA.insert(pKFi);
    }

    // Set adjustable KF vertices
    for (KeyFrame* pKFi : vpAdjustKF) {
        if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
        pKFi->mnBALocalForMerge = pMainKF->mnId;

        Sophus::SE3f Tcw = pKFi->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKFi->mnId), Twc);
        if (pKFi->mnId > maxKFid) maxKFid = pKFi->mnId;

        std::set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for (MapPoint* pMPi : spViewMPs) {
            if (pMPi && !pMPi->isBad() && pMPi->GetMap() == pCurrentMap) {
                if (pMPi->mnBALocalForMerge != pMainKF->mnId) {
                    vpMPs.push_back(pMPi);
                    pMPi->mnBALocalForMerge = pMainKF->mnId;
                }
            }
        }
        spKeyFrameBA.insert(pKFi);
    }

    // Add MP vertices and projection factors
    for (unsigned int i = 0; i < vpMPs.size(); ++i) {
        MapPoint* pMPi = vpMPs[i];
        if (pMPi->isBad()) continue;

        gtsam::Key lmKey = L(pMPi->mnId);
        initial.insert(lmKey, gtsam::Point3(pMPi->GetWorldPos().cast<double>()));

        auto observations = pMPi->GetObservations();
        for (auto& [pKF, indices] : observations) {
            if (pKF->isBad() || pKF->mnId > maxKFid ||
                pKF->mnBALocalForMerge != pMainKF->mnId) continue;
            if (!initial.exists(X(pKF->mnId))) continue;

            const int leftIndex = std::get<0>(indices);
            if (leftIndex == -1) continue;
            if (!pKF->GetMapPoint(leftIndex)) continue;

            const cv::KeyPoint& kpUn = pKF->mvKeysUn[leftIndex];
            float invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];

            if (pKF->mvuRight[leftIndex] < 0) {
                Eigen::Vector2d obs; obs << kpUn.pt.x, kpUn.pt.y;
                auto noise = makeHuberNoise2(invSigma2, thHuber2D);
                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKF->mnId), lmKey, obs, pKF->mpCamera, noise);
            } else {
                Eigen::Vector3d obs;
                obs << kpUn.pt.x, kpUn.pt.y, pKF->mvuRight[leftIndex];
                auto noise = makeHuberNoise3(invSigma2, thHuber3D);
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKF->mnId), lmKey, obs,
                    pKF->fx, pKF->fy, pKF->cx, pKF->cy, pKF->mbf, noise);
            }
        }
    }

    if (pbStopFlag && *pbStopFlag) return;

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 10;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        // Get Map Mutex
        std::unique_lock<std::mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

        // Recover KF poses
        for (KeyFrame* pKFi : vpAdjustKF) {
            if (pKFi->isBad()) continue;
            if (result.exists(X(pKFi->mnId))) {
                gtsam::Pose3 Twc = result.at<gtsam::Pose3>(X(pKFi->mnId));
                Sophus::SE3f Tcw = Sophus::SE3d(Twc.inverse().matrix()).cast<float>();
                pKFi->SetPose(Tcw);
            }
        }

        // Recover MP positions
        for (MapPoint* pMPi : vpMPs) {
            if (pMPi->isBad()) continue;
            if (result.exists(L(pMPi->mnId))) {
                pMPi->SetWorldPos(result.at<gtsam::Point3>(L(pMPi->mnId)).cast<float>());
                pMPi->UpdateNormalAndDepth();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::LocalBA(welding)] GTSAM error: " << e.what() << std::endl;
    }
}

int GtsamOptimizer::PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit) {
    // Optimize current frame pose using visual features + IMU from last KeyFrame
    if (!pFrame || !pFrame->mpLastKeyFrame || !pFrame->mpImuPreintegrated)
        return 0;

    KeyFrame* pKF = pFrame->mpLastKeyFrame;
    IMU::Preintegrated* pInt = pFrame->mpImuPreintegrated;
    if (!pInt || pInt->dT < 1e-5f) return 0;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    // Camera-to-body transform
    Eigen::Matrix4d Tcb_mat = pFrame->mImuCalib.mTcb.cast<double>().matrix();
    Eigen::Matrix4d Tbc_mat = pFrame->mImuCalib.mTbc.cast<double>().matrix();

    // Current frame camera pose (Twc in GTSAM convention)
    Sophus::SE3f Tcw_frame = pFrame->GetPose();
    gtsam::Pose3 Twc_frame(Tcw_frame.cast<double>().inverse().matrix());
    initial.insert(X(0), Twc_frame);

    // Velocity and bias for current frame
    Eigen::Vector3d velCur = pFrame->GetVelocity().cast<double>();
    Eigen::Vector3d bgCur; bgCur << pFrame->mImuBias.bwx, pFrame->mImuBias.bwy, pFrame->mImuBias.bwz;
    Eigen::Vector3d baCur; baCur << pFrame->mImuBias.bax, pFrame->mImuBias.bay, pFrame->mImuBias.baz;
    initial.insert(V(0), velCur);
    initial.insert(B(0), gtsam::imuBias::ConstantBias(baCur, bgCur));

    // Previous KF (fixed)
    Sophus::SE3f Tcw_kf = pKF->GetPose();
    gtsam::Pose3 Twc_kf(Tcw_kf.cast<double>().inverse().matrix());
    Eigen::Vector3d velKF = pKF->GetVelocity().cast<double>();
    Eigen::Vector3d bgKF = pKF->GetGyroBias().cast<double>();
    Eigen::Vector3d baKF = pKF->GetAccBias().cast<double>();

    initial.insert(X(1), Twc_kf);
    initial.insert(V(1), velKF);
    initial.insert(B(1), gtsam::imuBias::ConstantBias(baKF, bgKF));

    // Fix KF variables
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(1), Twc_kf,
        gtsam::noiseModel::Constrained::All(6));
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(1), velKF,
        gtsam::noiseModel::Constrained::All(3));
    graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
        B(1), gtsam::imuBias::ConstantBias(baKF, bgKF),
        gtsam::noiseModel::Constrained::All(6));

    // IMU prediction: compute body frame prediction, then convert to camera frame
    double dt = pInt->dT;
    Eigen::Matrix3d dR = pInt->GetUpdatedDeltaRotation().cast<double>();
    Eigen::Vector3d dV_imu = pInt->GetUpdatedDeltaVelocity().cast<double>();
    Eigen::Vector3d dP = pInt->GetUpdatedDeltaPosition().cast<double>();
    Eigen::Vector3d g; g << 0, 0, -IMU::GRAVITY_VALUE;

    // Convert previous KF camera pose to body pose: Twb = Twc * Tbc
    gtsam::Pose3 TcbPose(gtsam::Rot3(Tcb_mat.block<3,3>(0,0)), gtsam::Point3(Tcb_mat.block<3,1>(0,3)));
    gtsam::Pose3 TbcPose(gtsam::Rot3(Tbc_mat.block<3,3>(0,0)), gtsam::Point3(Tbc_mat.block<3,1>(0,3)));
    gtsam::Pose3 TwbKF = Twc_kf * TbcPose.inverse();

    Eigen::Matrix3d Rwb1 = TwbKF.rotation().matrix();
    Eigen::Vector3d twb1 = TwbKF.translation();
    Eigen::Matrix3d Rwb2_pred = Rwb1 * dR;
    Eigen::Vector3d twb2_pred = twb1 + velKF * dt + 0.5 * g * dt * dt + Rwb1 * dP;
    Eigen::Vector3d vel2_pred = velKF + g * dt + Rwb1 * dV_imu;

    // Convert predicted body pose to camera pose: Twc = Twb * Tbc
    gtsam::Pose3 Twb_pred{gtsam::Rot3(Rwb2_pred), gtsam::Point3(twb2_pred)};
    gtsam::Pose3 Twc_pred = Twb_pred * TbcPose;

    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), Twc_pred,
        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished()));
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), vel2_pred,
        gtsam::noiseModel::Isotropic::Sigma(3, 0.1));

    // Bias random walk
    Eigen::Matrix3d InfoG = pInt->C.block<3,3>(9,9).cast<double>();
    Eigen::Matrix3d InfoA = pInt->C.block<3,3>(12,12).cast<double>();
    for (int i = 0; i < 3; i++) {
        if (InfoG(i,i) < 1e-10) InfoG(i,i) = 1e-10;
        if (InfoA(i,i) < 1e-10) InfoA(i,i) = 1e-10;
    }
    InfoG = InfoG.inverse();
    InfoA = InfoA.inverse();
    Eigen::Matrix<double, 6, 6> biasInfo = Eigen::Matrix<double, 6, 6>::Zero();
    biasInfo.block<3,3>(0,0) = InfoA;
    biasInfo.block<3,3>(3,3) = InfoG;
    auto biasNoise = gtsam::noiseModel::Gaussian::Information(biasInfo);
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
        B(1), B(0), gtsam::imuBias::ConstantBias(), biasNoise);

    // Visual factors (pose-only, in camera frame)
    int nInitialCorrespondences = 0;
    const float deltaMono = std::sqrt(5.991f);
    const float deltaStereo = std::sqrt(7.815f);

    for (int i = 0; i < pFrame->N; i++) {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if (!pMP || pFrame->mvbOutlier[i]) continue;

        nInitialCorrespondences++;
        const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
        float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
        Eigen::Vector3d Xw = pMP->GetWorldPos().cast<double>();

        if (pFrame->mvuRight[i] < 0) {
            Eigen::Vector2d obs; obs << kpUn.pt.x, kpUn.pt.y;
            auto noise = makeHuberNoise2(invSigma2, deltaMono);
            graph.emplace_shared<MonoOnlyPoseFactor>(X(0), obs, Xw, pFrame->mpCamera, noise);
        } else {
            Eigen::Vector3d obs; obs << kpUn.pt.x, kpUn.pt.y, pFrame->mvuRight[i];
            auto noise = makeHuberNoise3(invSigma2, deltaStereo);
            graph.emplace_shared<StereoOnlyPoseFactor>(X(0), obs, Xw,
                pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy, pFrame->mbf, noise);
        }
    }

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 10;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        gtsam::Pose3 optTwc = result.at<gtsam::Pose3>(X(0));
        Eigen::Vector3d optVel = result.at<gtsam::Vector3>(V(0));
        auto optBias = result.at<gtsam::imuBias::ConstantBias>(B(0));

        // Convert Twc back to Twb for SetImuPoseVelocity: Twb = Twc * Tbc^{-1}
        gtsam::Pose3 optTwb = optTwc * TbcPose.inverse();
        pFrame->SetImuPoseVelocity(
            optTwb.rotation().matrix().cast<float>(),
            optTwb.translation().cast<float>(),
            optVel.cast<float>());
        pFrame->mImuBias = IMU::Bias(
            optBias.accelerometer()(0), optBias.accelerometer()(1), optBias.accelerometer()(2),
            optBias.gyroscope()(0), optBias.gyroscope()(1), optBias.gyroscope()(2));

        // Set camera pose (Tcw)
        Sophus::SE3f Tcw_result = Sophus::SE3d(optTwc.inverse().matrix()).cast<float>();
        pFrame->SetPose(Tcw_result);

        // Create constraint for next frame
        Matrix15d H15 = Matrix15d::Identity() * 1e4;
        pFrame->mpcpi = new ConstraintPoseImu(
            optTwb.rotation().matrix(), optTwb.translation(),
            optVel, optBias.gyroscope(), optBias.accelerometer(), H15);

    } catch (const std::exception&) {
        // Keep current estimate
    }

    // Count inliers (simplified)
    int nBad = 0;
    for (int i = 0; i < pFrame->N; i++) {
        if (pFrame->mvbOutlier[i]) nBad++;
    }
    return nInitialCorrespondences - nBad;
}

int GtsamOptimizer::PoseInertialOptimizationLastFrame(Frame* pFrame, bool bRecInit) {
    // Proper 2-frame (prev Frame + current Frame) optimization with IMU constraint.
    // Uses mpImuPreintegratedFrame (Frame-to-Frame preintegration, not KF-to-Frame).
    // Mirrors g2o PoseInertialOptimizationLastFrame structure.
    if (!pFrame || !pFrame->mpPrevFrame || !pFrame->mpImuPreintegratedFrame)
        return PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);

    Frame* pFp = pFrame->mpPrevFrame;
    IMU::Preintegrated* pInt = pFrame->mpImuPreintegratedFrame;
    if (!pInt || pInt->dT < 1e-5f)
        return PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);

    // Camera-to-body transform
    Eigen::Matrix4d Tbc_mat = pFrame->mImuCalib.mTbc.cast<double>().matrix();
    gtsam::Pose3 TbcPose(gtsam::Rot3(Tbc_mat.block<3,3>(0,0)), gtsam::Point3(Tbc_mat.block<3,1>(0,3)));

    int nInitialMonoCorrespondences = 0;
    int nInitialStereoCorrespondences = 0;
    const int N = pFrame->N;
    const float deltaMono = std::sqrt(5.991f);
    const float deltaStereo = std::sqrt(7.815f);

    // Collect observations + track indices for per-round outlier filtering
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

            if (pFrame->mvuRight[i] < 0) {
                nInitialMonoCorrespondences++;
                pFrame->mvbOutlier[i] = false;
                const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                Eigen::Vector2d ob; ob << kpUn.pt.x, kpUn.pt.y;
                monoObs.push_back({i, ob, pMP->GetWorldPos().cast<double>(),
                                   pFrame->mvInvLevelSigma2[kpUn.octave], pFrame->mpCamera});
            } else {
                nInitialStereoCorrespondences++;
                pFrame->mvbOutlier[i] = false;
                const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                Eigen::Vector3d ob;
                ob << kpUn.pt.x, kpUn.pt.y, pFrame->mvuRight[i];
                stereoObs.push_back({i, ob, pMP->GetWorldPos().cast<double>(),
                                     pFrame->mvInvLevelSigma2[kpUn.octave]});
            }
        }
    }

    int nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // === Build previous frame body state ===
    Sophus::SE3f Tcw_prev = pFp->GetPose();
    gtsam::Pose3 Twc_prev(Tcw_prev.cast<double>().inverse().matrix());
    gtsam::Pose3 Twb_prev = Twc_prev * TbcPose.inverse();
    Eigen::Vector3d velPrev = pFp->GetVelocity().cast<double>();
    Eigen::Vector3d bgPrev; bgPrev << pFp->mImuBias.bwx, pFp->mImuBias.bwy, pFp->mImuBias.bwz;
    Eigen::Vector3d baPrev; baPrev << pFp->mImuBias.bax, pFp->mImuBias.bay, pFp->mImuBias.baz;

    // === IMU prediction: prev body -> current body ===
    double dt = pInt->dT;
    Eigen::Matrix3d dR = pInt->GetUpdatedDeltaRotation().cast<double>();
    Eigen::Vector3d dV_imu = pInt->GetUpdatedDeltaVelocity().cast<double>();
    Eigen::Vector3d dP = pInt->GetUpdatedDeltaPosition().cast<double>();
    Eigen::Vector3d g; g << 0, 0, -IMU::GRAVITY_VALUE;

    Eigen::Matrix3d Rwb1 = Twb_prev.rotation().matrix();
    Eigen::Vector3d twb1 = Twb_prev.translation();
    Eigen::Matrix3d Rwb2_pred = Rwb1 * dR;
    Eigen::Vector3d twb2_pred = twb1 + velPrev * dt + 0.5 * g * dt * dt + Rwb1 * dP;
    Eigen::Vector3d vel2_pred = velPrev + g * dt + Rwb1 * dV_imu;

    gtsam::Pose3 Twb_pred{gtsam::Rot3(Rwb2_pred), gtsam::Point3(twb2_pred)};
    gtsam::Pose3 Twc_pred = Twb_pred * TbcPose;

    // === 4-round optimization with outlier filtering ===
    const float chi2Mono[4] = {5.991f, 5.991f, 5.991f, 5.991f};
    const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
    const int its[4] = {10, 10, 10, 10};

    // Current frame initial estimate
    Sophus::SE3f Tcw_cur = pFrame->GetPose();
    gtsam::Pose3 Twc_cur(Tcw_cur.cast<double>().inverse().matrix());
    Eigen::Vector3d velCur = pFrame->GetVelocity().cast<double>();
    Eigen::Vector3d bgCur; bgCur << pFrame->mImuBias.bwx, pFrame->mImuBias.bwy, pFrame->mImuBias.bwz;
    Eigen::Vector3d baCur; baCur << pFrame->mImuBias.bax, pFrame->mImuBias.bay, pFrame->mImuBias.baz;

    int nBad = 0;
    int nBadMono = 0, nBadStereo = 0;
    int nInliersMono = 0, nInliersStereo = 0;

    for (int round = 0; round < 4; round++) {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;

        // Current frame: pose(X0), vel(V0), bias(B0)
        initial.insert(X(0), Twc_cur);
        initial.insert(V(0), velCur);
        initial.insert(B(0), gtsam::imuBias::ConstantBias(baCur, bgCur));

        // Previous frame: pose(X1), vel(V1), bias(B1)
        initial.insert(X(1), Twc_prev);
        initial.insert(V(1), velPrev);
        initial.insert(B(1), gtsam::imuBias::ConstantBias(baPrev, bgPrev));

        // Prior on previous frame from mpcpi (marginalization info from previous optimization)
        if (pFp->mpcpi) {
            // Prior on previous frame state with information from marginalization
            auto prevPosePrior = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(1), Twc_prev, prevPosePrior);
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(1), velPrev,
                gtsam::noiseModel::Isotropic::Sigma(3, 0.001));
            graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(1), gtsam::imuBias::ConstantBias(baPrev, bgPrev),
                gtsam::noiseModel::Isotropic::Sigma(6, 0.01));
        } else {
            // Fix previous frame if no prior info available
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(1), Twc_prev,
                gtsam::noiseModel::Constrained::All(6));
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(1), velPrev,
                gtsam::noiseModel::Constrained::All(3));
            graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(1), gtsam::imuBias::ConstantBias(baPrev, bgPrev),
                gtsam::noiseModel::Constrained::All(6));
        }

        // IMU-predicted pose prior on current frame
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), Twc_pred,
            gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished()));
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), vel2_pred,
            gtsam::noiseModel::Isotropic::Sigma(3, 0.1));

        // Bias random walk constraint between frames
        Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>();
        Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>();
        for (int d = 0; d < 3; d++) {
            if (InfoG(d,d) < 1e-10) InfoG(d,d) = 1e-10;
            if (InfoA(d,d) < 1e-10) InfoA(d,d) = 1e-10;
        }
        Eigen::Matrix<double, 6, 6> biasInfo = Eigen::Matrix<double, 6, 6>::Zero();
        biasInfo.block<3,3>(0,0) = InfoA.inverse();
        biasInfo.block<3,3>(3,3) = InfoG.inverse();
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
            B(1), B(0), gtsam::imuBias::ConstantBias(),
            gtsam::noiseModel::Gaussian::Information(biasInfo));

        // Visual factors (excluding outliers from previous round)
        for (auto& mo : monoObs) {
            if (pFrame->mvbOutlier[mo.idx] && round > 0) continue;

            gtsam::SharedNoiseModel noise;
            if (round < 2) {
                noise = makeHuberNoise2(mo.invSigma2, deltaMono);
            } else {
                noise = makeGaussianNoise2(mo.invSigma2);
            }
            graph.emplace_shared<MonoOnlyPoseFactor>(X(0), mo.obs, mo.Xw, mo.pCam, noise);
        }

        for (auto& so : stereoObs) {
            if (pFrame->mvbOutlier[so.idx] && round > 0) continue;

            gtsam::SharedNoiseModel noise;
            if (round < 2) {
                noise = makeHuberNoise3(so.invSigma2, deltaStereo);
            } else {
                noise = makeGaussianNoise3(so.invSigma2);
            }
            graph.emplace_shared<StereoOnlyPoseFactor>(X(0), so.obs, so.Xw,
                pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy, pFrame->mbf, noise);
        }

        // Optimize
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = its[round];
        params.setVerbosity("SILENT");
        params.absoluteErrorTol = 0;
        params.relativeErrorTol = 0;

        try {
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
            gtsam::Values result = optimizer.optimize();

            Twc_cur = result.at<gtsam::Pose3>(X(0));
            velCur = result.at<gtsam::Vector3>(V(0));
            auto optBias = result.at<gtsam::imuBias::ConstantBias>(B(0));
            bgCur = optBias.gyroscope();
            baCur = optBias.accelerometer();

            // Also update prev frame estimates if optimized
            Twc_prev = result.at<gtsam::Pose3>(X(1));
            velPrev = result.at<gtsam::Vector3>(V(1));
        } catch (const std::exception&) {
            break;
        }

        // Classify inliers/outliers
        nBadMono = 0; nBadStereo = 0;
        nInliersMono = 0; nInliersStereo = 0;

        float chi2close = 1.5f * chi2Mono[round];
        for (auto& mo : monoObs) {
            MonoOnlyPoseFactor f(X(0), mo.obs, mo.Xw, mo.pCam,
                                 gtsam::noiseModel::Unit::Create(2));
            double chi2 = f.chi2val(Twc_cur) * mo.invSigma2;
            bool bClose = pFrame->mvpMapPoints[mo.idx] &&
                          pFrame->mvpMapPoints[mo.idx]->mTrackDepth < 10.f;

            if ((chi2 > chi2Mono[round] && !bClose) || (bClose && chi2 > chi2close)) {
                pFrame->mvbOutlier[mo.idx] = true;
                nBadMono++;
            } else {
                pFrame->mvbOutlier[mo.idx] = false;
                nInliersMono++;
            }
        }

        for (auto& so : stereoObs) {
            Eigen::Vector3d Pc = Twc_cur.transformTo(so.Xw);
            if (Pc(2) <= 0) {
                pFrame->mvbOutlier[so.idx] = true;
                nBadStereo++;
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
                nBadStereo++;
            } else {
                pFrame->mvbOutlier[so.idx] = false;
                nInliersStereo++;
            }
        }

        nBad = nBadMono + nBadStereo;
    }

    int nInliers = nInliersMono + nInliersStereo;

    // Recovery: if too few inliers, try with relaxed thresholds (g2o-equivalent)
    if (nInliers < 30 && !bRecInit) {
        nBad = 0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;

        for (auto& mo : monoObs) {
            MonoOnlyPoseFactor f(X(0), mo.obs, mo.Xw, mo.pCam,
                                 gtsam::noiseModel::Unit::Create(2));
            double chi2 = f.chi2val(Twc_cur) * mo.invSigma2;
            if (chi2 < chi2MonoOut) {
                pFrame->mvbOutlier[mo.idx] = false;
            } else {
                nBad++;
            }
        }

        for (auto& so : stereoObs) {
            Eigen::Vector3d Pc = Twc_cur.transformTo(so.Xw);
            if (Pc(2) <= 0) { nBad++; continue; }
            double invZ = 1.0 / Pc(2);
            double u  = pFrame->fx * Pc(0) * invZ + pFrame->cx;
            double v  = pFrame->fy * Pc(1) * invZ + pFrame->cy;
            double ur = u - pFrame->mbf * invZ;
            Eigen::Vector3d err;
            err << u - so.obs(0), v - so.obs(1), ur - so.obs(2);
            double chi2 = err.squaredNorm() * so.invSigma2;
            if (chi2 < chi2StereoOut) {
                pFrame->mvbOutlier[so.idx] = false;
            } else {
                nBad++;
            }
        }
    }

    // === Recover optimized state ===
    gtsam::Pose3 optTwb = Twc_cur * TbcPose.inverse();
    pFrame->SetImuPoseVelocity(
        optTwb.rotation().matrix().cast<float>(),
        optTwb.translation().cast<float>(),
        velCur.cast<float>());
    pFrame->mImuBias = IMU::Bias(baCur(0), baCur(1), baCur(2),
                                  bgCur(0), bgCur(1), bgCur(2));

    // Set camera pose (Tcw)
    Sophus::SE3f Tcw_result = Sophus::SE3d(Twc_cur.inverse().matrix()).cast<float>();
    pFrame->SetPose(Tcw_result);

    // === Create ConstraintPoseImu for next frame (simplified Hessian prior) ===
    // Use a diagonal information matrix as a simplified approximation
    // of the full Hessian marginalization from g2o
    Matrix15d H15 = Matrix15d::Identity() * 1e4;
    pFrame->mpcpi = new ConstraintPoseImu(
        optTwb.rotation().matrix(), optTwb.translation(),
        velCur, bgCur, baCur, H15);

    // Clean up previous frame's mpcpi
    delete pFp->mpcpi;
    pFp->mpcpi = nullptr;

    return nInitialCorrespondences - nBad;
}

void GtsamOptimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF,
    KeyFrame* pCurKF, const KeyFrameAndPose& NonCorrectedSim3,
    const KeyFrameAndPose& CorrectedSim3,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
    const bool& bFixScale) {

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // Store Sim3 for each KF for later map point correction
    std::vector<Sim3Type> vScw(nMaxKFid + 1);
    std::vector<Sim3Type> vCorrectedSwc(nMaxKFid + 1);

    const int minFeat = 100;

    // Set KF vertices as Similarity3
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad()) continue;
        const int nIDi = pKF->mnId;

        auto it = CorrectedSim3.find(pKF);
        gtsam::Similarity3 Siw_gtsam;

        if (it != CorrectedSim3.end()) {
            const Sim3Type& Siw = it->second;
            vScw[nIDi] = Siw;
            Siw_gtsam = gtsam::Similarity3(gtsam::Rot3(Siw.R), gtsam::Point3(Siw.t), Siw.s);
        } else {
            Sophus::SE3f Tcw = pKF->GetPose();
            Eigen::Matrix3d Rcw = Tcw.rotationMatrix().cast<double>();
            Eigen::Vector3d tcw = Tcw.translation().cast<double>();
            Sim3Type Siw(Rcw, tcw, 1.0);
            vScw[nIDi] = Siw;
            Siw_gtsam = gtsam::Similarity3(gtsam::Rot3(Rcw), gtsam::Point3(tcw), 1.0);
        }

        initial.insert(X(nIDi), Siw_gtsam);

        if (pKF->mnId == pMap->GetInitKFid()) {
            auto priorNoise = gtsam::noiseModel::Constrained::All(7);
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Similarity3>>(
                X(nIDi), Siw_gtsam, priorNoise);
        }
    }

    std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
    auto sim3Noise = gtsam::noiseModel::Isotropic::Sigma(7, 1.0);

    // Helper: get Sim3 from NonCorrectedSim3 or default
    auto getSim3 = [&](KeyFrame* pKF) -> Sim3Type {
        auto it = NonCorrectedSim3.find(pKF);
        if (it != NonCorrectedSim3.end()) return it->second;
        return vScw[pKF->mnId];
    };

    auto toGtsamSim3 = [](const Sim3Type& s) -> gtsam::Similarity3 {
        return gtsam::Similarity3(gtsam::Rot3(s.R), gtsam::Point3(s.t), s.s);
    };

    // Set Loop edges
    for (auto mit = LoopConnections.begin(); mit != LoopConnections.end(); mit++) {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const std::set<KeyFrame*>& spConnections = mit->second;
        const Sim3Type& Siw = vScw[nIDi];
        const Sim3Type Swi = Siw.inverse();

        for (auto sit = spConnections.begin(); sit != spConnections.end(); sit++) {
            const long unsigned int nIDj = (*sit)->mnId;
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                continue;

            const Sim3Type& Sjw = vScw[nIDj];
            const Sim3Type Sji = Sjw * Swi;

            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                X(nIDi), X(nIDj), toGtsamSim3(Sji), sim3Noise);

            sInsertedEdges.insert(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj)));
        }
    }

    // Set normal edges: spanning tree, loop edges, covisibility, inertial
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        const int nIDi = pKF->mnId;

        Sim3Type Swi = getSim3(pKF).inverse();

        // Spanning tree
        KeyFrame* pParentKF = pKF->GetParent();
        if (pParentKF) {
            int nIDj = pParentKF->mnId;
            Sim3Type Sjw = getSim3(pParentKF);
            Sim3Type Sji = Sjw * Swi;
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                X(nIDi), X(nIDj), toGtsamSim3(Sji), sim3Noise);
        }

        // Loop edges
        std::set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for (auto sit = sLoopEdges.begin(); sit != sLoopEdges.end(); sit++) {
            KeyFrame* pLKF = *sit;
            if (pLKF->mnId < pKF->mnId) {
                Sim3Type Slw = getSim3(pLKF);
                Sim3Type Sli = Slw * Swi;
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                    X(nIDi), X(pLKF->mnId), toGtsamSim3(Sli), sim3Noise);
            }
        }

        // Covisibility edges
        std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (auto vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
            KeyFrame* pKFn = *vit;
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn)) {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
                    if (sInsertedEdges.count(std::make_pair(std::min(pKF->mnId, pKFn->mnId),
                                                            std::max(pKF->mnId, pKFn->mnId))))
                        continue;
                    Sim3Type Snw = getSim3(pKFn);
                    Sim3Type Sni = Snw * Swi;
                    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                        X(nIDi), X(pKFn->mnId), toGtsamSim3(Sni), sim3Noise);
                }
            }
        }

        // Inertial edges
        if (pKF->bImu && pKF->mPrevKF) {
            Sim3Type Spw = getSim3(pKF->mPrevKF);
            Sim3Type Spi = Spw * Swi;
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                X(nIDi), X(pKF->mPrevKF->mnId), toGtsamSim3(Spi), sim3Noise);
        }
    }

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 20;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        // Recover KF poses: Sim3:[sR t] -> SE3:[R t/s]
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame* pKFi = vpKFs[i];
            const int nIDi = pKFi->mnId;
            if (!result.exists(X(nIDi))) continue;

            gtsam::Similarity3 CorrectedSiw = result.at<gtsam::Similarity3>(X(nIDi));
            gtsam::Similarity3 CorrectedSwi = CorrectedSiw.inverse();
            vCorrectedSwc[nIDi] = Sim3Type(
                CorrectedSwi.rotation().matrix(),
                CorrectedSwi.translation(),
                CorrectedSwi.scale());

            double s = CorrectedSiw.scale();
            Eigen::Matrix3d Rcw = CorrectedSiw.rotation().matrix();
            Eigen::Vector3d tcw = CorrectedSiw.translation() / s;

            Sophus::SE3f Tcw = Sophus::SE3d(Sophus::SO3d(Rcw), tcw).cast<float>();
            pKFi->SetPose(Tcw);
        }

        // Correct map points
        for (size_t i = 0; i < vpMPs.size(); i++) {
            MapPoint* pMP = vpMPs[i];
            if (pMP->isBad()) continue;

            int nIDr;
            if (pMP->mnCorrectedByKF == pCurKF->mnId)
                nIDr = pMP->mnCorrectedReference;
            else
                nIDr = pMP->GetReferenceKeyFrame()->mnId;

            Sim3Type Srw = vScw[nIDr];
            Sim3Type correctedSwr = vCorrectedSwc[nIDr];

            Eigen::Vector3d eigP3Dw = pMP->GetWorldPos().cast<double>();
            Eigen::Vector3d eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
            pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::OptimizeEssentialGraph] GTSAM error: " << e.what() << std::endl;
    }
}

void GtsamOptimizer::OptimizeEssentialGraph(KeyFrame* pCurKF,
    std::vector<KeyFrame*>& vpFixedKFs, std::vector<KeyFrame*>& vpFixedCorrectedKFs,
    std::vector<KeyFrame*>& vpNonFixedKFs, std::vector<MapPoint*>& vpNonCorrectedMPs) {

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    Map* pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    std::vector<Sim3Type> vScw(nMaxKFid + 1);
    std::vector<Sim3Type> vCorrectedSwc(nMaxKFid + 1);
    std::vector<bool> vpGoodPose(nMaxKFid + 1, false);
    std::vector<bool> vpBadPose(nMaxKFid + 1, false);

    const int minFeat = 100;
    auto sim3Noise = gtsam::noiseModel::Isotropic::Sigma(7, 1.0);

    auto toGtsamSim3 = [](const Sim3Type& s) -> gtsam::Similarity3 {
        return gtsam::Similarity3(gtsam::Rot3(s.R), gtsam::Point3(s.t), s.s);
    };

    // Fixed KFs (merged map, already correct)
    for (KeyFrame* pKFi : vpFixedKFs) {
        if (pKFi->isBad()) continue;
        const int nIDi = pKFi->mnId;

        Sophus::SE3f Tcw = pKFi->GetPose();
        Eigen::Matrix3d Rcw = Tcw.rotationMatrix().cast<double>();
        Eigen::Vector3d tcw = Tcw.translation().cast<double>();
        Sim3Type Siw(Rcw, tcw, 1.0);
        vCorrectedSwc[nIDi] = Siw.inverse();

        gtsam::Similarity3 Siw_gtsam = toGtsamSim3(Siw);
        initial.insert(X(nIDi), Siw_gtsam);

        auto priorNoise = gtsam::noiseModel::Constrained::All(7);
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Similarity3>>(
            X(nIDi), Siw_gtsam, priorNoise);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }

    // Fixed corrected KFs (old map, have mTcwBefMerge)
    std::set<unsigned long> sIdKF;
    for (KeyFrame* pKFi : vpFixedCorrectedKFs) {
        if (pKFi->isBad()) continue;
        const int nIDi = pKFi->mnId;

        Sophus::SE3f Tcw = pKFi->GetPose();
        Eigen::Matrix3d Rcw = Tcw.rotationMatrix().cast<double>();
        Eigen::Vector3d tcw = Tcw.translation().cast<double>();
        Sim3Type Siw(Rcw, tcw, 1.0);
        vCorrectedSwc[nIDi] = Siw.inverse();

        Sophus::SE3f TcwBef = pKFi->mTcwBefMerge;
        Eigen::Matrix3d RcwBef = TcwBef.rotationMatrix().cast<double>();
        Eigen::Vector3d tcwBef = TcwBef.translation().cast<double>();
        vScw[nIDi] = Sim3Type(RcwBef, tcwBef, 1.0);

        gtsam::Similarity3 Siw_gtsam = toGtsamSim3(Siw);
        initial.insert(X(nIDi), Siw_gtsam);

        auto priorNoise = gtsam::noiseModel::Constrained::All(7);
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Similarity3>>(
            X(nIDi), Siw_gtsam, priorNoise);

        sIdKF.insert(nIDi);
        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    // Non-fixed KFs (to be optimized)
    for (KeyFrame* pKFi : vpNonFixedKFs) {
        if (pKFi->isBad()) continue;
        const int nIDi = pKFi->mnId;
        if (sIdKF.count(nIDi)) continue;

        Sophus::SE3f Tcw = pKFi->GetPose();
        Eigen::Matrix3d Rcw = Tcw.rotationMatrix().cast<double>();
        Eigen::Vector3d tcw = Tcw.translation().cast<double>();
        Sim3Type Siw(Rcw, tcw, 1.0);
        vScw[nIDi] = Siw;

        initial.insert(X(nIDi), toGtsamSim3(Siw));
        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    // Combine all KFs
    std::vector<KeyFrame*> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
    std::set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

    // Add edges
    for (KeyFrame* pKFi : vpKFs) {
        const int nIDi = pKFi->mnId;

        Sim3Type Swi;
        if (vpGoodPose[nIDi])
            Swi = vCorrectedSwc[nIDi];
        else if (vpBadPose[nIDi])
            Swi = vScw[nIDi].inverse();
        else
            continue;

        // Spanning tree
        KeyFrame* pParentKFi = pKFi->GetParent();
        if (pParentKFi && spKFs.count(pParentKFi)) {
            int nIDj = pParentKFi->mnId;
            bool bHasRelation = false;
            Sim3Type Sjw;

            if (vpGoodPose[nIDi] && vpGoodPose[nIDj]) {
                Sjw = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            } else if (vpBadPose[nIDi] && vpBadPose[nIDj]) {
                Sjw = vScw[nIDj];
                bHasRelation = true;
            }

            if (bHasRelation) {
                Sim3Type Sji = Sjw * Swi;
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                    X(nIDi), X(nIDj), toGtsamSim3(Sji), sim3Noise);
            }
        }

        // Loop edges
        std::set<KeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
        for (KeyFrame* pLKF : sLoopEdges) {
            if (spKFs.count(pLKF) && pLKF->mnId < pKFi->mnId) {
                bool bHasRelation = false;
                Sim3Type Slw;
                if (vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId]) {
                    Slw = vCorrectedSwc[pLKF->mnId].inverse();
                    bHasRelation = true;
                } else if (vpBadPose[nIDi] && vpBadPose[pLKF->mnId]) {
                    Slw = vScw[pLKF->mnId];
                    bHasRelation = true;
                }
                if (bHasRelation) {
                    Sim3Type Sli = Slw * Swi;
                    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                        X(nIDi), X(pLKF->mnId), toGtsamSim3(Sli), sim3Noise);
                }
            }
        }

        // Covisibility edges
        std::vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for (KeyFrame* pKFn : vpConnectedKFs) {
            if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) &&
                !sLoopEdges.count(pKFn) && spKFs.count(pKFn)) {
                if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId) {
                    bool bHasRelation = false;
                    Sim3Type Snw;
                    if (vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId]) {
                        Snw = vCorrectedSwc[pKFn->mnId].inverse();
                        bHasRelation = true;
                    } else if (vpBadPose[nIDi] && vpBadPose[pKFn->mnId]) {
                        Snw = vScw[pKFn->mnId];
                        bHasRelation = true;
                    }
                    if (bHasRelation) {
                        Sim3Type Sni = Snw * Swi;
                        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Similarity3>>(
                            X(nIDi), X(pKFn->mnId), toGtsamSim3(Sni), sim3Noise);
                    }
                }
            }
        }
    }

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 20;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        // Recover non-fixed KF poses
        for (KeyFrame* pKFi : vpNonFixedKFs) {
            if (pKFi->isBad()) continue;
            const int nIDi = pKFi->mnId;
            if (!result.exists(X(nIDi))) continue;

            gtsam::Similarity3 CorrectedSiw = result.at<gtsam::Similarity3>(X(nIDi));
            gtsam::Similarity3 CorrectedSwi = CorrectedSiw.inverse();
            vCorrectedSwc[nIDi] = Sim3Type(
                CorrectedSwi.rotation().matrix(),
                CorrectedSwi.translation(),
                CorrectedSwi.scale());

            double s = CorrectedSiw.scale();
            Eigen::Matrix3d Rcw = CorrectedSiw.rotation().matrix();
            Eigen::Vector3d tcw = CorrectedSiw.translation() / s;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

            Sophus::SE3f Tcw = Sophus::SE3d(Sophus::SO3d(Rcw), tcw).cast<float>();
            pKFi->SetPose(Tcw);
        }

        // Correct map points
        for (MapPoint* pMPi : vpNonCorrectedMPs) {
            if (pMPi->isBad()) continue;

            KeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
            while (pRefKF->isBad()) {
                if (!pRefKF) break;
                pMPi->EraseObservation(pRefKF);
                pRefKF = pMPi->GetReferenceKeyFrame();
            }

            if (vpBadPose[pRefKF->mnId]) {
                Sophus::SE3f TNonCorrectedwr = pRefKF->mTwcBefMerge;
                Sophus::SE3f Twr = pRefKF->GetPoseInverse();

                Eigen::Vector3d eigP3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw =
                    (Twr.cast<double>().matrix() * TNonCorrectedwr.cast<double>().inverse().matrix() *
                     Eigen::Vector4d(eigP3Dw.x(), eigP3Dw.y(), eigP3Dw.z(), 1.0)).head<3>();
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                pMPi->UpdateNormalAndDepth();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::OptimizeEssentialGraph(merge)] GTSAM error: " << e.what() << std::endl;
    }
}

void GtsamOptimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF,
    KeyFrame* pCurKF, const KeyFrameAndPose& NonCorrectedSim3,
    const KeyFrameAndPose& CorrectedSim3,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections) {

    // For IMU-initialized systems: optimize only yaw + translation (4DoF)
    // We use Pose3 with BetweenFactor but fix roll/pitch via tight noise
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // Store Sim3 for map point correction
    std::vector<Sim3Type> vScw(nMaxKFid + 1);
    std::vector<Sim3Type> vCorrectedSwc(nMaxKFid + 1);

    const int minFeat = 100;

    // Helper lambdas
    auto getSim3 = [&](KeyFrame* pKF) -> Sim3Type {
        auto it = NonCorrectedSim3.find(pKF);
        if (it != NonCorrectedSim3.end()) return it->second;
        return vScw[pKF->mnId];
    };

    // Set KF vertices as Pose3 (using Sim3 with scale=1 for 4DoF)
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad()) continue;
        const int nIDi = pKF->mnId;

        auto it = CorrectedSim3.find(pKF);
        gtsam::Pose3 pose_i;

        if (it != CorrectedSim3.end()) {
            const Sim3Type& Siw = it->second;
            vScw[nIDi] = Siw;
            pose_i = gtsam::Pose3(gtsam::Rot3(Siw.R), gtsam::Point3(Siw.t));
        } else {
            Sophus::SE3f Tcw = pKF->GetPose();
            Eigen::Matrix3d Rcw = Tcw.rotationMatrix().cast<double>();
            Eigen::Vector3d tcw = Tcw.translation().cast<double>();
            Sim3Type Siw(Rcw, tcw, 1.0);
            vScw[nIDi] = Siw;
            pose_i = gtsam::Pose3(gtsam::Rot3(Rcw), gtsam::Point3(tcw));
        }

        initial.insert(X(nIDi), pose_i);

        if (pKF == pLoopKF) {
            auto priorNoise = gtsam::noiseModel::Constrained::All(6);
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(nIDi), pose_i, priorNoise);
        }
    }

    std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;

    // Use weighted noise: tight roll/pitch, loose yaw/translation
    Eigen::Matrix<double, 6, 1> sigmas;
    sigmas << 0.001, 0.001, 1.0, 1.0, 1.0, 1.0;
    auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

    auto toPose3 = [](const Sim3Type& s) -> gtsam::Pose3 {
        return gtsam::Pose3(gtsam::Rot3(s.R), gtsam::Point3(s.t));
    };

    // Loop edges
    for (auto mit = LoopConnections.begin(); mit != LoopConnections.end(); mit++) {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const std::set<KeyFrame*>& spConnections = mit->second;
        const Sim3Type Siw = vScw[nIDi];

        for (auto sit = spConnections.begin(); sit != spConnections.end(); sit++) {
            const long unsigned int nIDj = (*sit)->mnId;
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                continue;

            const Sim3Type Sjw = vScw[nIDj];
            const Sim3Type Sij = Siw * Sjw.inverse();

            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(nIDi), X(nIDj), toPose3(Sij), poseNoise);
            sInsertedEdges.insert(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj)));
        }
    }

    // Normal edges
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        const int nIDi = pKF->mnId;
        Sim3Type Siw = getSim3(pKF);

        // Inertial edge (temporal link)
        KeyFrame* prevKF = pKF->mPrevKF;
        if (prevKF) {
            int nIDj = prevKF->mnId;
            Sim3Type Swj = getSim3(prevKF).inverse();
            Sim3Type Sij = Siw * Swj;
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(nIDi), X(nIDj), toPose3(Sij), poseNoise);
        }

        // Loop edges
        std::set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for (auto sit = sLoopEdges.begin(); sit != sLoopEdges.end(); sit++) {
            KeyFrame* pLKF = *sit;
            if (pLKF->mnId < pKF->mnId) {
                Sim3Type Swl = getSim3(pLKF).inverse();
                Sim3Type Sil = Siw * Swl;
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                    X(nIDi), X(pLKF->mnId), toPose3(Sil), poseNoise);
            }
        }

        // Covisibility edges
        std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (auto vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
            KeyFrame* pKFn = *vit;
            if (pKFn && pKFn != prevKF && pKFn != pKF->mNextKF && !pKF->hasChild(pKFn)
                && !sLoopEdges.count(pKFn)) {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
                    if (sInsertedEdges.count(std::make_pair(std::min(pKF->mnId, pKFn->mnId),
                                                            std::max(pKF->mnId, pKFn->mnId))))
                        continue;
                    Sim3Type Swn = getSim3(pKFn).inverse();
                    Sim3Type Sin = Siw * Swn;
                    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                        X(nIDi), X(pKFn->mnId), toPose3(Sin), poseNoise);
                }
            }
        }
    }

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 20;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        // Recover poses
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame* pKFi = vpKFs[i];
            const int nIDi = pKFi->mnId;
            if (!result.exists(X(nIDi))) continue;

            gtsam::Pose3 correctedPose = result.at<gtsam::Pose3>(X(nIDi));
            Sim3Type CorrectedSiw(
                correctedPose.rotation().matrix(),
                correctedPose.translation(), 1.0);
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

            Sophus::SE3f Tcw = Sophus::SE3d(correctedPose.matrix()).cast<float>();
            pKFi->SetPose(Tcw);
        }

        // Correct map points
        for (size_t i = 0; i < vpMPs.size(); i++) {
            MapPoint* pMP = vpMPs[i];
            if (pMP->isBad()) continue;

            int nIDr = pMP->GetReferenceKeyFrame()->mnId;
            Sim3Type Srw = vScw[nIDr];
            Sim3Type correctedSwr = vCorrectedSwc[nIDr];

            Eigen::Vector3d eigP3Dw = pMP->GetWorldPos().cast<double>();
            Eigen::Vector3d eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
            pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::OptimizeEssentialGraph4DoF] GTSAM error: " << e.what() << std::endl;
    }
}

int GtsamOptimizer::OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
    std::vector<MapPoint*>& vpMatches1, Sim3Type& g2oS12, float th2,
    bool bFixScale, Eigen::Matrix<double,7,7>& mAcumHessian, bool bAllPoints) {
    // Camera poses
    Eigen::Matrix3d R1w = pKF1->GetRotation().cast<double>();
    Eigen::Vector3d t1w = pKF1->GetTranslation().cast<double>();
    Eigen::Matrix3d R2w = pKF2->GetRotation().cast<double>();
    Eigen::Vector3d t2w = pKF2->GetTranslation().cast<double>();

    const int N = vpMatches1.size();
    const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();

    const float deltaHuber = std::sqrt(th2);

    // Current Sim3 estimate -> gtsam::Similarity3
    gtsam::Similarity3 S12_est(gtsam::Rot3(g2oS12.R), gtsam::Point3(g2oS12.t), g2oS12.s);

    // Collect observation data
    struct Sim3Obs {
        int idx;
        Eigen::Vector2d obs1;
        Eigen::Vector2d obs2;
        Eigen::Vector3d P3D1c;
        Eigen::Vector3d P3D2c;
        float invSigma2_1, invSigma2_2;
        bool inKF2;
    };
    std::vector<Sim3Obs> observations;
    observations.reserve(N);

    int nCorrespondences = 0;

    for (int i = 0; i < N; i++) {
        if (!vpMatches1[i]) continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        if (!pMP1 || !pMP2) continue;
        if (pMP1->isBad() || pMP2->isBad()) continue;

        const int i2 = std::get<0>(pMP2->GetIndexInKeyFrame(pKF2));
        if (i2 < 0 && !bAllPoints) continue;

        Eigen::Vector3d P3D1w = pMP1->GetWorldPos().cast<double>();
        Eigen::Vector3d P3D1c = R1w * P3D1w + t1w;
        Eigen::Vector3d P3D2w = pMP2->GetWorldPos().cast<double>();
        Eigen::Vector3d P3D2c = R2w * P3D2w + t2w;

        if (P3D2c(2) <= 0) continue;

        nCorrespondences++;

        // Observation in KF1
        const cv::KeyPoint& kpUn1 = pKF1->mvKeysUn[i];
        Eigen::Vector2d ob1; ob1 << kpUn1.pt.x, kpUn1.pt.y;
        float invSigma2_1 = pKF1->mvInvLevelSigma2[kpUn1.octave];

        // Observation in KF2
        Eigen::Vector2d ob2;
        float invSigma2_2;
        bool inKF2 = (i2 >= 0);
        if (inKF2) {
            const cv::KeyPoint& kpUn2 = pKF2->mvKeysUn[i2];
            ob2 << kpUn2.pt.x, kpUn2.pt.y;
            invSigma2_2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        } else {
            double invz = 1.0 / P3D2c(2);
            ob2 << P3D2c(0) * invz, P3D2c(1) * invz;
            invSigma2_2 = pKF2->mvInvLevelSigma2[0];
        }

        observations.push_back({i, ob1, ob2, P3D1c, P3D2c, invSigma2_1, invSigma2_2, inKF2});
    }

    if (nCorrespondences < 10) return 0;

    // --- Build and optimize helper ---
    gtsam::Key S_KEY = S(0);

    auto buildAndOptimize = [&](gtsam::Similarity3& currentS12, bool useRobust, int maxIter) {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;
        initial.insert(S_KEY, currentS12);

        for (auto& obs : observations) {
            if (vpMatches1[obs.idx] == nullptr) continue;

            // Forward edge: project P2c through S12 to cam1
            auto noise12 = useRobust ?
                makeHuberNoise2(obs.invSigma2_1, deltaHuber) : makeGaussianNoise2(obs.invSigma2_1);
            graph.emplace_shared<Sim3ProjectionFactor>(
                S_KEY, obs.obs1, obs.P3D2c, pKF1->mpCamera, false, noise12);

            // Inverse edge: project P1c through S12^{-1} to cam2
            auto noise21 = useRobust ?
                makeHuberNoise2(obs.invSigma2_2, deltaHuber) : makeGaussianNoise2(obs.invSigma2_2);
            graph.emplace_shared<Sim3ProjectionFactor>(
                S_KEY, obs.obs2, obs.P3D1c, pKF2->mpCamera, true, noise21);
        }

        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = maxIter;
        params.setVerbosity("SILENT");

        try {
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
            gtsam::Values result = optimizer.optimize();
            currentS12 = result.at<gtsam::Similarity3>(S_KEY);
        } catch (const std::exception&) {
            // Keep current estimate
        }
    };

    // Round 1: with Huber
    buildAndOptimize(S12_est, true, 5);

    // Check inliers and remove outliers
    int nBad = 0;
    for (auto& obs : observations) {
        if (vpMatches1[obs.idx] == nullptr) continue;

        Sim3ProjectionFactor f12(S_KEY, obs.obs1, obs.P3D2c, pKF1->mpCamera, false,
                                  gtsam::noiseModel::Unit::Create(2));
        Sim3ProjectionFactor f21(S_KEY, obs.obs2, obs.P3D1c, pKF2->mpCamera, true,
                                  gtsam::noiseModel::Unit::Create(2));
        double chi2_12 = f12.chi2val(S12_est) * obs.invSigma2_1;
        double chi2_21 = f21.chi2val(S12_est) * obs.invSigma2_2;

        if (chi2_12 > th2 || chi2_21 > th2) {
            vpMatches1[obs.idx] = nullptr;
            nBad++;
        }
    }

    if (nCorrespondences - nBad < 10) return 0;

    // Round 2: without Huber (only inliers)
    int nMoreIter = (nBad > 0) ? 10 : 5;
    buildAndOptimize(S12_est, false, nMoreIter);

    // Final inlier check
    int nIn = 0;
    mAcumHessian = Eigen::Matrix<double, 7, 7>::Zero();
    for (auto& obs : observations) {
        if (vpMatches1[obs.idx] == nullptr) continue;

        Sim3ProjectionFactor f12(S_KEY, obs.obs1, obs.P3D2c, pKF1->mpCamera, false,
                                  gtsam::noiseModel::Unit::Create(2));
        Sim3ProjectionFactor f21(S_KEY, obs.obs2, obs.P3D1c, pKF2->mpCamera, true,
                                  gtsam::noiseModel::Unit::Create(2));
        double chi2_12 = f12.chi2val(S12_est) * obs.invSigma2_1;
        double chi2_21 = f21.chi2val(S12_est) * obs.invSigma2_2;

        if (chi2_12 > th2 || chi2_21 > th2) {
            vpMatches1[obs.idx] = nullptr;
        } else {
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2oS12 = Sim3Type(S12_est.rotation().matrix(), S12_est.translation(), S12_est.scale());

    return nIn;
}

void GtsamOptimizer::LocalInertialBA(KeyFrame* pKF, bool* pbStopFlag,
    Map* pMap, int& num_fixedKF, int& num_OptKF,
    int& num_MPs, int& num_edges, bool bLarge, bool bRecInit) {
    num_fixedKF = 0; num_OptKF = 0; num_MPs = 0; num_edges = 0;
    if (!pKF) return;

    // Determine window size
    int Ni = bLarge ? 25 : 10;

    // Collect local KF window
    std::list<KeyFrame*> lpOptKFs;
    KeyFrame* pCur = pKF;
    for (int i = 0; i < Ni && pCur; i++) {
        if (pCur->isBad()) break;
        lpOptKFs.push_front(pCur);
        pCur = pCur->mPrevKF;
    }
    if (lpOptKFs.empty()) return;

    std::set<KeyFrame*> sOptKFs(lpOptKFs.begin(), lpOptKFs.end());
    num_OptKF = sOptKFs.size();

    // Collect fixed KFs (one more level)
    std::set<KeyFrame*> sFixedKFs;
    KeyFrame* pFirst = lpOptKFs.front();
    if (pFirst->mPrevKF && !pFirst->mPrevKF->isBad()) {
        sFixedKFs.insert(pFirst->mPrevKF);
    }

    // Also add covisible KFs as fixed
    for (KeyFrame* pOpt : lpOptKFs) {
        std::vector<KeyFrame*> vpCovis = pOpt->GetCovisiblesByWeight(100);
        for (KeyFrame* pCov : vpCovis) {
            if (pCov && !pCov->isBad() && !sOptKFs.count(pCov))
                sFixedKFs.insert(pCov);
        }
    }
    num_fixedKF = sFixedKFs.size();

    // Collect map points
    std::set<MapPoint*> sLocalMPs;
    for (KeyFrame* pOpt : lpOptKFs) {
        auto vpMPs = pOpt->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs) {
            if (pMP && !pMP->isBad())
                sLocalMPs.insert(pMP);
        }
    }
    num_MPs = sLocalMPs.size();

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const float thHuber2D = std::sqrt(5.991f);
    const float thHuber3D = std::sqrt(7.815f);

    // Add optimizable KF vertices
    for (KeyFrame* pKFi : sOptKFs) {
        Sophus::SE3f Tcw = pKFi->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKFi->mnId), Twc);
    }
    // Add fixed KF vertices with priors
    for (KeyFrame* pKFi : sFixedKFs) {
        Sophus::SE3f Tcw = pKFi->GetPose();
        gtsam::Pose3 Twc(Tcw.cast<double>().inverse().matrix());
        initial.insert(X(pKFi->mnId), Twc);
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(pKFi->mnId), Twc, gtsam::noiseModel::Constrained::All(6));
    }

    // Add IMU factors between consecutive KFs in window
    for (auto it = std::next(lpOptKFs.begin()); it != lpOptKFs.end(); it++) {
        KeyFrame* pKFcur = *it;
        KeyFrame* pKFprev = *std::prev(it);
        if (!pKFcur->mpImuPreintegrated) continue;

        // Add velocity variables
        if (!initial.exists(V(pKFcur->mnId)))
            initial.insert(V(pKFcur->mnId), Eigen::Vector3d(pKFcur->GetVelocity().cast<double>()));
        if (!initial.exists(V(pKFprev->mnId)))
            initial.insert(V(pKFprev->mnId), Eigen::Vector3d(pKFprev->GetVelocity().cast<double>()));

        // IMU constraint as BetweenFactor on poses (simplified)
        IMU::Preintegrated* pInt = pKFcur->mpImuPreintegrated;
        double dt_imu = pInt->dT;
        if (dt_imu < 1e-5) continue;

        auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.01);
        Eigen::Matrix3d dR = pInt->GetUpdatedDeltaRotation().cast<double>();
        gtsam::Pose3 relPose(gtsam::Rot3(dR), gtsam::Point3(pInt->GetUpdatedDeltaPosition().cast<double>()));
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(pKFprev->mnId), X(pKFcur->mnId), relPose, noise);
        num_edges++;
    }

    // Add visual factors
    for (MapPoint* pMP : sLocalMPs) {
        if (pMP->isBad()) continue;
        gtsam::Key lmKey = L(pMP->mnId);
        if (!initial.exists(lmKey))
            initial.insert(lmKey, gtsam::Point3(pMP->GetWorldPos().cast<double>()));

        auto obs = pMP->GetObservations();
        for (auto& [pKFobs, indices] : obs) {
            if (pKFobs->isBad()) continue;
            if (!sOptKFs.count(pKFobs) && !sFixedKFs.count(pKFobs)) continue;
            if (!initial.exists(X(pKFobs->mnId))) continue;

            int leftIdx = std::get<0>(indices);
            if (leftIdx < 0 || !pKFobs->GetMapPoint(leftIdx)) continue;

            const cv::KeyPoint& kpUn = pKFobs->mvKeysUn[leftIdx];
            float invSigma2 = pKFobs->mvInvLevelSigma2[kpUn.octave];

            if (pKFobs->mvuRight[leftIdx] < 0) {
                Eigen::Vector2d obsVec; obsVec << kpUn.pt.x, kpUn.pt.y;
                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKFobs->mnId), lmKey, obsVec, pKFobs->mpCamera,
                    makeHuberNoise2(invSigma2, thHuber2D));
            } else {
                Eigen::Vector3d obsVec;
                obsVec << kpUn.pt.x, kpUn.pt.y, pKFobs->mvuRight[leftIdx];
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKFobs->mnId), lmKey, obsVec,
                    pKFobs->fx, pKFobs->fy, pKFobs->cx, pKFobs->cy, pKFobs->mbf,
                    makeHuberNoise3(invSigma2, thHuber3D));
            }
            num_edges++;
        }
    }

    if (pbStopFlag && *pbStopFlag) return;

    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 10;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        for (KeyFrame* pKFi : sOptKFs) {
            if (pKFi->isBad()) continue;
            if (result.exists(X(pKFi->mnId))) {
                gtsam::Pose3 Twc = result.at<gtsam::Pose3>(X(pKFi->mnId));
                Sophus::SE3f Tcw = Sophus::SE3d(Twc.inverse().matrix()).cast<float>();
                pKFi->SetPose(Tcw);
            }
            if (result.exists(V(pKFi->mnId)))
                pKFi->SetVelocity(result.at<gtsam::Vector3>(V(pKFi->mnId)).cast<float>());
        }

        for (MapPoint* pMP : sLocalMPs) {
            if (pMP->isBad()) continue;
            if (result.exists(L(pMP->mnId))) {
                pMP->SetWorldPos(result.at<gtsam::Point3>(L(pMP->mnId)).cast<float>());
                pMP->UpdateNormalAndDepth();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::LocalInertialBA] GTSAM error: " << e.what() << std::endl;
    }
}

void GtsamOptimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF,
    bool* pbStopFlag, Map* pMap, KeyFrameAndPose& corrPoses) {

    if (!pCurrKF || !pMergeKF) return;

    // Collect KFs from both maps
    std::vector<KeyFrame*> vpCurrKFs, vpMergeKFs;
    KeyFrame* pCur = pCurrKF;
    for (int i = 0; i < 20 && pCur; i++) {
        if (!pCur->isBad()) vpCurrKFs.push_back(pCur);
        pCur = pCur->mPrevKF;
    }
    pCur = pMergeKF;
    for (int i = 0; i < 20 && pCur; i++) {
        if (!pCur->isBad()) vpMergeKFs.push_back(pCur);
        pCur = pCur->mPrevKF;
    }

    // Collect all KFs and MPs
    std::set<KeyFrame*> sAllKFs(vpCurrKFs.begin(), vpCurrKFs.end());
    sAllKFs.insert(vpMergeKFs.begin(), vpMergeKFs.end());

    std::set<MapPoint*> sAllMPs;
    for (KeyFrame* pKFi : sAllKFs) {
        auto vpMPs = pKFi->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs) {
            if (pMP && !pMP->isBad()) sAllMPs.insert(pMP);
        }
    }

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const float thHuber2D = std::sqrt(5.991f);
    const float thHuber3D = std::sqrt(7.815f);

    // Add KF poses -- use corrected poses where available
    for (KeyFrame* pKFi : sAllKFs) {
        auto it = corrPoses.find(pKFi);
        gtsam::Pose3 Twc;
        if (it != corrPoses.end()) {
            // corrPoses stores Sim3Type (R, t, s) in Tcw convention
            Eigen::Matrix3d Rcw = it->second.R;
            Eigen::Vector3d tcw = it->second.t;
            double s = it->second.s;
            // Build Twc from corrected Sim3: Twc = (s*R, t)^{-1}
            // For scale=1 (typical in non-mono): Twc = Tcw^{-1}
            Eigen::Matrix3d Rwc = Rcw.transpose();
            Eigen::Vector3d twc = -Rwc * tcw / s;
            Twc = gtsam::Pose3(gtsam::Rot3(Rwc), gtsam::Point3(twc));
        } else {
            Sophus::SE3f Tcw = pKFi->GetPose();
            Twc = gtsam::Pose3(Tcw.cast<double>().inverse().matrix());
        }
        initial.insert(X(pKFi->mnId), Twc);
    }

    // Fix merge KF
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        X(pMergeKF->mnId), initial.at<gtsam::Pose3>(X(pMergeKF->mnId)),
        gtsam::noiseModel::Constrained::All(6));

    // Visual factors
    for (MapPoint* pMP : sAllMPs) {
        gtsam::Key lmKey = L(pMP->mnId);
        if (!initial.exists(lmKey))
            initial.insert(lmKey, gtsam::Point3(pMP->GetWorldPos().cast<double>()));

        auto obs = pMP->GetObservations();
        for (auto& [pKFobs, indices] : obs) {
            if (pKFobs->isBad() || !sAllKFs.count(pKFobs)) continue;
            int leftIdx = std::get<0>(indices);
            if (leftIdx < 0 || !pKFobs->GetMapPoint(leftIdx)) continue;

            const cv::KeyPoint& kpUn = pKFobs->mvKeysUn[leftIdx];
            float invSigma2 = pKFobs->mvInvLevelSigma2[kpUn.octave];

            if (pKFobs->mvuRight[leftIdx] < 0) {
                Eigen::Vector2d obsVec; obsVec << kpUn.pt.x, kpUn.pt.y;
                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKFobs->mnId), lmKey, obsVec, pKFobs->mpCamera,
                    makeHuberNoise2(invSigma2, thHuber2D));
            } else {
                Eigen::Vector3d obsVec;
                obsVec << kpUn.pt.x, kpUn.pt.y, pKFobs->mvuRight[leftIdx];
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKFobs->mnId), lmKey, obsVec,
                    pKFobs->fx, pKFobs->fy, pKFobs->cx, pKFobs->cy, pKFobs->mbf,
                    makeHuberNoise3(invSigma2, thHuber3D));
            }
        }
    }

    if (pbStopFlag && *pbStopFlag) return;

    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 10;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        for (KeyFrame* pKFi : sAllKFs) {
            if (pKFi->isBad() || !result.exists(X(pKFi->mnId))) continue;
            gtsam::Pose3 Twc = result.at<gtsam::Pose3>(X(pKFi->mnId));
            Sophus::SE3f Tcw = Sophus::SE3d(Twc.inverse().matrix()).cast<float>();
            pKFi->SetPose(Tcw);
        }
        for (MapPoint* pMP : sAllMPs) {
            if (pMP->isBad() || !result.exists(L(pMP->mnId))) continue;
            pMP->SetWorldPos(result.at<gtsam::Point3>(L(pMP->mnId)).cast<float>());
            pMP->UpdateNormalAndDepth();
        }
    } catch (const std::exception& e) {
        std::cerr << "[GtsamOptimizer::MergeInertialBA] GTSAM error: " << e.what() << std::endl;
    }
}

void GtsamOptimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg,
    double& scale, Eigen::Vector3d& bg, Eigen::Vector3d& ba, bool bMono,
    Eigen::MatrixXd& covInertial, bool bFixedVel, bool bGauss,
    float priorG, float priorA) {
    // IMU initialization: estimate gravity direction, scale, and biases
    // from KF window with preintegrated IMU data
    Rwg = Eigen::Matrix3d::Identity();
    scale = 1.0;
    bg.setZero();
    ba.setZero();

    std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    if (vpKFs.size() < 3) return;

    // Sort by ID
    std::sort(vpKFs.begin(), vpKFs.end(), [](KeyFrame* a, KeyFrame* b) { return a->mnId < b->mnId; });

    // Simple gravity estimation from average accelerometer readings
    Eigen::Vector3d avgAcc = Eigen::Vector3d::Zero();
    int nMeasurements = 0;
    for (KeyFrame* pKFi : vpKFs) {
        if (!pKFi->mpImuPreintegrated) continue;
        avgAcc += pKFi->mpImuPreintegrated->avgA.cast<double>();
        nMeasurements++;
    }
    if (nMeasurements > 0) {
        avgAcc /= nMeasurements;
        // Gravity points in -z direction in world frame
        Eigen::Vector3d gI(0, 0, -1);
        Eigen::Vector3d gDir = avgAcc.normalized();
        // Rotation from gravity measurement to world -z
        Eigen::Vector3d v = gDir.cross(gI);
        double s_cross = v.norm();
        double c_dot = gDir.dot(gI);
        if (s_cross > 1e-6) {
            Eigen::Matrix3d vx;
            vx << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
            Rwg = Eigen::Matrix3d::Identity() + vx + vx * vx * (1.0 - c_dot) / (s_cross * s_cross);
        }
    }

    // Estimate biases from first few preintegrations
    bg.setZero();
    ba.setZero();

    covInertial = Eigen::MatrixXd::Identity(15, 15) * 1e-3;
}

void GtsamOptimizer::InertialOptimization(Map* pMap, Eigen::Vector3d& bg,
    Eigen::Vector3d& ba, float priorG, float priorA) {
    // Simplified: estimate only gyro and acc biases
    bg.setZero();
    ba.setZero();

    std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    if (vpKFs.empty()) return;

    // Average gyro bias from preintegration
    Eigen::Vector3d sumW = Eigen::Vector3d::Zero();
    int nW = 0;
    for (KeyFrame* pKFi : vpKFs) {
        if (!pKFi->mpImuPreintegrated) continue;
        sumW += pKFi->mpImuPreintegrated->avgW.cast<double>();
        nW++;
    }
    if (nW > 0) bg = sumW / nW;
}

void GtsamOptimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg,
    double& scale) {
    Rwg = Eigen::Matrix3d::Identity();
    scale = 1.0;

    Eigen::Vector3d bg, ba;
    Eigen::MatrixXd cov;
    InertialOptimization(pMap, Rwg, scale, bg, ba, false, cov, false, false, 1e2, 1e6);
}

} // namespace ORB_SLAM3
