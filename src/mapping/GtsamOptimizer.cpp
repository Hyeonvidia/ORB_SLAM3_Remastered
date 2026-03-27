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
#include "Optimizer.hpp"
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
#include "G2oTypes.hpp"  // for ConstraintPoseImu, Matrix15d, EdgeInertialGS, etc.

// g2o headers for InertialOptimization (uses same g2o factor graph as G2oOptimizer)
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/core/robust_kernel_impl.h"

#include <boost/optional.hpp>
#include <mutex>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ORB_SLAM3
{

// Helper function to safely construct Sophus::SE3f without triggering isOrthogonal asserts from float casting
static Sophus::SE3f toSophus(const gtsam::Pose3& pose) {
    Eigen::Quaternionf q(pose.rotation().toQuaternion().cast<float>());
    q.normalize(); // Guarantee perfect strict orthogonality
    return Sophus::SE3f(q, pose.translation().cast<float>());
}

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

        if (std::abs(Pc(2)) < 1e-6) Pc(2) = (Pc(2) >= 0) ? 1e-6 : -1e-6;

        Eigen::Vector2d proj = pCamera_->project(Pc);
        gtsam::Vector2 error = proj - measured_;

        if (H) {
            Eigen::Matrix<double, 2, 3> Jproj = pCamera_->projectJac(Pc);
            *H = Jproj * Hpose;
        }

        return error;
    }

    /// Compute squared UNROBUSTIFIED error for exact classical outlier classification
    double chi2val(const gtsam::Pose3& Twc) const {
        Eigen::Vector3d Pc = Twc.transformTo(Xw_);
        if (std::abs(Pc(2)) < 1e-6) Pc(2) = (Pc(2) >= 0) ? 1e-6 : -1e-6;
        Eigen::Vector2d proj = pCamera_->project(Pc);
        return (proj - measured_).squaredNorm();
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

        if (std::abs(Pc(2)) < 1e-6) Pc(2) = (Pc(2) >= 0) ? 1e-6 : -1e-6;

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

        if (std::abs(Pc(2)) < 1e-6) Pc(2) = (Pc(2) >= 0) ? 1e-6 : -1e-6;

        Eigen::Vector2d proj = pCamera_->project(Pc);
        gtsam::Vector2 error = proj - measured_;

        if (H1 || H2) {
            Eigen::Matrix<double, 2, 3> Jproj = pCamera_->projectJac(Pc);
            if (H1) *H1 = Jproj * Hpose;
            if (H2) *H2 = Jproj * Hpoint;
        }

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

        if (std::abs(Pc(2)) < 1e-6) Pc(2) = (Pc(2) >= 0) ? 1e-6 : -1e-6;

        double invZ = 1.0 / Pc(2);
        double u = fx_ * Pc(0) * invZ + cx_;
        double v = fy_ * Pc(1) * invZ + cy_;
        double ur = u - bf_ * invZ;

        gtsam::Vector3 error;
        error << u - measured_(0), v - measured_(1), ur - measured_(2);

        if (H1 || H2) {
            double invZ2 = invZ * invZ;
            Eigen::Matrix<double, 3, 3> Jproj;
            Jproj << fx_ * invZ, 0, -fx_ * Pc(0) * invZ2,
                     0, fy_ * invZ, -fy_ * Pc(1) * invZ2,
                     fx_ * invZ, 0, -(fx_ * Pc(0) - bf_) * invZ2;
            if (H1) *H1 = Jproj * Hpose;
            if (H2) *H2 = Jproj * Hpoint;
        }

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
// Use GTSAM's built-in noiseModel::Robust with mEstimator::Huber.
// This applies the Huber loss holistically during optimization (matching g2o),
// rather than manually altering error/Jacobians inside evaluateError.
static gtsam::SharedNoiseModel makeHuberNoise2(double invSigma2, double delta) {
    double sigma = 1.0 / std::sqrt(invSigma2);
    auto base = gtsam::noiseModel::Isotropic::Sigma(2, sigma);
    auto huber = gtsam::noiseModel::mEstimator::Huber::Create(delta);
    return gtsam::noiseModel::Robust::Create(huber, base);
}

static gtsam::SharedNoiseModel makeHuberNoise3(double invSigma2, double delta) {
    double sigma = 1.0 / std::sqrt(invSigma2);
    auto base = gtsam::noiseModel::Isotropic::Sigma(3, sigma);
    auto huber = gtsam::noiseModel::mEstimator::Huber::Create(delta);
    return gtsam::noiseModel::Robust::Create(huber, base);
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
        params.diagonalDamping = true;
        params.lambdaInitial = 100.0;
        params.lambdaUpperBound = 1e9;  // Prevent premature LM exit
        params.relativeErrorTol = 1e-5;
        params.absoluteErrorTol = 1e-5;
        params.maxIterations = its[round];
        params.setVerbosity("SILENT");
        params.absoluteErrorTol = 0;
        params.relativeErrorTol = 0;
        try {
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
            gtsam::Values result = optimizer.optimize();
            currentPose = result.at<gtsam::Pose3>(X(0));
            // Convert Twc (GTSAM) back to Tcw (ORB-SLAM3) via Sophus
            Sophus::SE3f Tcw_result = toSophus(currentPose.inverse());
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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;  // Prevent premature LM exit
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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

            Sophus::SE3f Tcw = toSophus(optimizedTwc.inverse());

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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;  // Prevent premature LM exit
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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
            Sophus::SE3f Tcw = toSophus(optimizedTwc.inverse());

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

    // Match g2o: count InitKF as a fixed KF
    for (auto* pKFi : lLocalKeyFrames) {
        if (pKFi->mnId == pMap->GetInitKFid()) {
            num_fixedKF++;
            break;
        }
    }

    if (lLocalKeyFrames.size() < 2) return;

    // Match g2o: abort if no fixed KFs (no anchor for the graph)
    if (num_fixedKF == 0) return;

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

        // Match g2o: fix InitKF with Constrained prior (anchor for gauge freedom)
        if (pKFi->mnId == pMap->GetInitKFid()) {
            graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                X(pKFi->mnId), Twc, gtsam::noiseModel::Constrained::All(6));
        }
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
                    X(pKFi->mnId), lmKey, ob, pKFi->mpCamera, noise, thHuber2D);
                vEdgeInfo.push_back({pKFi, pMP, true, invSigma2, factorIdx});
            } else {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Vector3d ob;
                ob << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise3(invSigma2, thHuber3D);
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKFi->mnId), lmKey, ob,
                    pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy, pKFi->mbf, noise, thHuber3D);
                vEdgeInfo.push_back({pKFi, pMP, false, invSigma2, factorIdx});
            }
            edgeCount++;
        }
    }
    num_edges = edgeCount;

    // === Single-round LBA (matches g2o: optimize(10) with Huber kernel) ===
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 10;
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;  // Prevent premature LM exit (match g2o)
    // Match g2o: use high initial lambda for inertial mode (more conservative)
    if (pMap->IsInertial())
        params.lambdaInitial = 100.0;

    gtsam::Values result;
    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        result = optimizer.optimize();
    } catch (const std::exception& e) {
        std::cerr << "[LBA] GTSAM error: " << e.what() << std::endl;
        return;
    }

    // Classify outliers and collect observations to erase (g2o-equivalent)
    // Use RAW chi2 (unrobustified) matching g2o's e->chi2() + e->isDepthPositive()
    std::vector<std::pair<KeyFrame*,MapPoint*>> vToErase;
    for (auto& ei : vEdgeInfo) {
        if (ei.pMP->isBad()) continue;

        if (!result.exists(X(ei.pKF->mnId)) || !result.exists(L(ei.pMP->mnId)))
            continue;

        gtsam::Pose3 Twc = result.at<gtsam::Pose3>(X(ei.pKF->mnId));
        gtsam::Point3 Xw = result.at<gtsam::Point3>(L(ei.pMP->mnId));
        Eigen::Vector3d Pc = Twc.transformTo(Xw);

        // Check depth positivity (matches g2o isDepthPositive)
        if (Pc(2) <= 0.0) {
            vToErase.push_back(std::make_pair(ei.pKF, ei.pMP));
            continue;
        }

        // Compute raw chi2 manually
        if (ei.isMono) {
            Eigen::Vector2d proj = ei.pKF->mpCamera->project(Pc);
            const int leftIndex = std::get<0>(ei.pMP->GetIndexInKeyFrame(ei.pKF));
            if (leftIndex < 0) continue;
            const cv::KeyPoint& kpUn = ei.pKF->mvKeysUn[leftIndex];
            Eigen::Vector2d obs; obs << kpUn.pt.x, kpUn.pt.y;
            double rawChi2 = (proj - obs).squaredNorm() * ei.invSigma2;
            if (rawChi2 > 5.991)
                vToErase.push_back(std::make_pair(ei.pKF, ei.pMP));
        } else {
            double invZ = 1.0 / Pc(2);
            double u = ei.pKF->fx * Pc(0) * invZ + ei.pKF->cx;
            double v = ei.pKF->fy * Pc(1) * invZ + ei.pKF->cy;
            double ur = u - ei.pKF->mbf * invZ;
            const int leftIndex = std::get<0>(ei.pMP->GetIndexInKeyFrame(ei.pKF));
            if (leftIndex < 0) continue;
            const cv::KeyPoint& kpUn = ei.pKF->mvKeysUn[leftIndex];
            Eigen::Vector3d obs; obs << kpUn.pt.x, kpUn.pt.y, ei.pKF->mvuRight[leftIndex];
            Eigen::Vector3d pred; pred << u, v, ur;
            double rawChi2 = (pred - obs).squaredNorm() * ei.invSigma2;
            if (rawChi2 > 7.815)
                vToErase.push_back(std::make_pair(ei.pKF, ei.pMP));
        }
    }

    // Compute shifts BEFORE writing back to check for catastrophic results
    double totalPoseShift = 0;
    int nPoseRecovered = 0;
    std::vector<std::pair<KeyFrame*, gtsam::Pose3>, Eigen::aligned_allocator<std::pair<KeyFrame*, gtsam::Pose3>>> poseUpdates;
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

    // Apply pose updates (Twc -> Tcw via Sophus)
    for (auto& [pKFi, newTwc] : poseUpdates) {
        Sophus::SE3f Tcw = toSophus(newTwc.inverse());
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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;  // Prevent premature LM exit
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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
                Sophus::SE3f Tcw = toSophus(Twc.inverse());
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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;  // Prevent premature LM exit
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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
        Sophus::SE3f Tcw_result = toSophus(optTwc.inverse());
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
        params.diagonalDamping = true;
        params.lambdaInitial = 100.0;
        params.lambdaUpperBound = 1e9;
        params.relativeErrorTol = 0;
        params.absoluteErrorTol = 0;
        params.maxIterations = its[round];
        params.setVerbosity("SILENT");

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
    Sophus::SE3f Tcw_result = toSophus(Twc_cur.inverse());
    pFrame->SetPose(Tcw_result);

    // === Create ConstraintPoseImu via Hessian marginalization (matching g2o) ===
    // Build the final round's graph with only inlier visual factors for Hessian extraction
    // Variable ordering (g2o convention):
    //   [0-5]:   prev pose     [6-8]:  prev vel     [9-11]:  prev bg    [12-14]: prev ba
    //   [15-20]: cur pose      [21-23]: cur vel      [24-26]: cur bg     [27-29]: cur ba
    Matrix15d H15;
    try {
        // Reconstruct clean graph for Hessian: IMU + bias + prior + inlier visual factors
        gtsam::NonlinearFactorGraph hGraph;
        gtsam::Values hValues;

        hValues.insert(X(0), Twc_cur);
        hValues.insert(V(0), velCur);
        hValues.insert(B(0), gtsam::imuBias::ConstantBias(baCur, bgCur));
        hValues.insert(X(1), Twc_prev);
        hValues.insert(V(1), velPrev);
        hValues.insert(B(1), gtsam::imuBias::ConstantBias(baPrev, bgPrev));

        // Prior on previous frame (same as last optimization round)
        if (pFp->mpcpi) {
            hGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(1), Twc_prev,
                gtsam::noiseModel::Diagonal::Sigmas(
                    (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished()));
            hGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(1), velPrev,
                gtsam::noiseModel::Isotropic::Sigma(3, 0.001));
            hGraph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(1), gtsam::imuBias::ConstantBias(baPrev, bgPrev),
                gtsam::noiseModel::Isotropic::Sigma(6, 0.01));
        } else {
            hGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(1), Twc_prev,
                gtsam::noiseModel::Constrained::All(6));
            hGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(1), velPrev,
                gtsam::noiseModel::Constrained::All(3));
            hGraph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(1), gtsam::imuBias::ConstantBias(baPrev, bgPrev),
                gtsam::noiseModel::Constrained::All(6));
        }

        // IMU prediction prior
        hGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), Twc_pred,
            gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished()));
        hGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), vel2_pred,
            gtsam::noiseModel::Isotropic::Sigma(3, 0.1));

        // Bias random walk
        Eigen::Matrix3d InfoGh = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>();
        Eigen::Matrix3d InfoAh = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>();
        for (int d = 0; d < 3; d++) {
            if (InfoGh(d,d) < 1e-10) InfoGh(d,d) = 1e-10;
            if (InfoAh(d,d) < 1e-10) InfoAh(d,d) = 1e-10;
        }
        Eigen::Matrix<double, 6, 6> bInfoH = Eigen::Matrix<double, 6, 6>::Zero();
        bInfoH.block<3,3>(0,0) = InfoAh.inverse();
        bInfoH.block<3,3>(3,3) = InfoGh.inverse();
        hGraph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
            B(1), B(0), gtsam::imuBias::ConstantBias(),
            gtsam::noiseModel::Gaussian::Information(bInfoH));

        // Inlier visual factors only (Gaussian noise, no robust kernel)
        for (auto& mo : monoObs) {
            if (pFrame->mvbOutlier[mo.idx]) continue;
            auto noise = makeGaussianNoise2(mo.invSigma2);
            hGraph.emplace_shared<MonoOnlyPoseFactor>(X(0), mo.obs, mo.Xw, mo.pCam, noise);
        }
        for (auto& so : stereoObs) {
            if (pFrame->mvbOutlier[so.idx]) continue;
            auto noise = makeGaussianNoise3(so.invSigma2);
            hGraph.emplace_shared<StereoOnlyPoseFactor>(X(0), so.obs, so.Xw,
                pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy, pFrame->mbf, noise);
        }

        // Linearize and extract Hessian with specific variable ordering
        // Order: X(1), V(1), B(1), X(0), V(0), B(0)
        // This maps to g2o layout: prev_state(15) | cur_state(15)
        gtsam::Ordering ordering;
        ordering.push_back(X(1));  // prev pose [0-5]
        ordering.push_back(V(1));  // prev vel  [6-8]
        ordering.push_back(B(1));  // prev bias [9-14]  (6D: acc+gyro)
        ordering.push_back(X(0));  // cur pose  [15-20]
        ordering.push_back(V(0));  // cur vel   [21-23]
        ordering.push_back(B(0));  // cur bias  [24-29]

        auto linearGraph = hGraph.linearize(hValues);
        auto augmentedH = linearGraph->augmentedHessian(ordering);

        // augmentedH is (n+1) x (n+1) with the last row/col being the RHS
        // We need the top-left 30x30 as our Hessian
        int totalDim = 6 + 3 + 6 + 6 + 3 + 6;  // 30
        Eigen::Matrix<double, 30, 30> H30 = augmentedH.block(0, 0, totalDim, totalDim);

        // Marginalize previous frame states (rows/cols 0-14) using Schur complement
        Eigen::MatrixXd Hmarginalized = Optimizer::Marginalize(H30, 0, 14);

        // Extract 15x15 block for current frame
        H15 = Hmarginalized.block<15,15>(15, 15);

        // Ensure symmetric and positive semi-definite
        H15 = 0.5 * (H15 + H15.transpose());

        // Clamp extreme eigenvalues for numerical stability
        Eigen::SelfAdjointEigenSolver<Matrix15d> eigSolver(H15);
        if (eigSolver.info() == Eigen::Success) {
            Eigen::Matrix<double, 15, 1> eigVals = eigSolver.eigenvalues();
            bool needFix = false;
            for (int i = 0; i < 15; i++) {
                if (eigVals(i) < 0) { eigVals(i) = 0; needFix = true; }
            }
            if (needFix) {
                H15 = eigSolver.eigenvectors() * eigVals.asDiagonal()
                      * eigSolver.eigenvectors().transpose();
            }
        }
    } catch (const std::exception&) {
        // Fallback to simplified diagonal prior if Hessian extraction fails
        H15 = Matrix15d::Identity() * 1e4;
    }

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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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
            Eigen::Quaterniond qRcw(CorrectedSiw.rotation().toQuaternion());
            qRcw.normalize();
            Eigen::Vector3d tcw = CorrectedSiw.translation() / s;

            Sophus::SE3f Tcw = Sophus::SE3d(qRcw, tcw).cast<float>();
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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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
            Eigen::Quaterniond qRcw(CorrectedSiw.rotation().toQuaternion());
            qRcw.normalize();
            Eigen::Vector3d tcw = CorrectedSiw.translation() / s;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

            Sophus::SE3f Tcw = Sophus::SE3d(qRcw, tcw).cast<float>();
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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
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

            Sophus::SE3f Tcw = toSophus(correctedPose);
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
        params.diagonalDamping = true;
        params.lambdaInitial = 100.0;
        params.lambdaUpperBound = 1e9;
        params.relativeErrorTol = 1e-5;
        params.absoluteErrorTol = 1e-5;
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
    // -------------------------------------------------------------------------
    // Full g2o-based Local Inertial BA (identical to G2oOptimizer).
    // Uses EdgeInertial, EdgeGyroRW, EdgeAccRW, per-KF bias, visual factors.
    // -------------------------------------------------------------------------
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt = 10;
    int opt_it = 10;
    if (bLarge) {
        maxOpt = 25;
        opt_it = 4;
    }
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap() - 2, maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    std::vector<KeyFrame*> vpOptimizableKFs;
    const std::vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    std::list<KeyFrame*> lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for (int i = 1; i < Nd; i++) {
        if (vpOptimizableKFs.back()->mPrevKF) {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        } else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    std::list<MapPoint*> lLocalMapPoints;
    for (int i = 0; i < N; i++) {
        std::vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for (auto vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
            MapPoint* pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId) {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe
    std::list<KeyFrame*> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->mPrevKF) {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
    } else {
        vpOptimizableKFs.back()->mnBALocalForKF = 0;
        vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for (int i = 0, iend = vpNeighsKFs.size(); i < iend; i++) {
        if (lpOptVisKFs.size() >= maxCovKF) break;
        KeyFrame* pKFi = vpNeighsKFs[i];
        if (pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
            lpOptVisKFs.push_back(pKFi);
            std::vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for (auto vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
                MapPoint* pMP = *vit;
                if (pMP)
                    if (!pMP->isBad())
                        if (pMP->mnBALocalForKF != pKF->mnId) {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs from map point observations
    const int maxFixKF = 200;
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        std::map<KeyFrame*, std::tuple<int, int>> observations = (*lit)->GetObservations();
        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad()) {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if (lFixedKeyFrames.size() >= maxFixKF) break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup g2o optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    if (bLarge) {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-2);
        optimizer.setAlgorithm(solver);
    } else {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }

    // Set Local temporal KeyFrame vertices
    N = vpOptimizableKFs.size();
    num_OptKF = N;
    for (int i = 0; i < N; i++) {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        VertexPose* VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if (pKFi->bImu) {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual KF vertices
    for (auto it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++) {
        KeyFrame* pKFi = *it;
        VertexPose* VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed KeyFrame vertices
    num_fixedKF = lFixedKeyFrames.size();
    for (auto lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        VertexPose* VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if (pKFi->bImu) {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create inertial constraints
    std::vector<EdgeInertial*> vei(N, (EdgeInertial*)nullptr);
    std::vector<EdgeGyroRW*> vegr(N, (EdgeGyroRW*)nullptr);
    std::vector<EdgeAccRW*> vear(N, (EdgeAccRW*)nullptr);

    for (int i = 0; i < N; i++) {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        if (!pKFi->mPrevKF) {
            std::cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << std::endl;
            continue;
        }
        if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated) {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);

            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2) {
                std::cerr << "Error building inertial edge" << std::endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);
            vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            if (i == N - 1 || bRecInit) {
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if (i == N - 1)
                    vei[i]->setInformation(vei[i]->information() * 1e-2);
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0, VG1);
            vegr[i]->setVertex(1, VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0, VA1);
            vear[i]->setVertex(1, VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        } else
            std::cout << "ERROR building inertial edge" << std::endl;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (N + lFixedKeyFrames.size()) * lLocalMapPoints.size();

    std::vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    std::vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    std::vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    std::vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    std::vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    std::vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid * 5;
    num_MPs = lLocalMapPoints.size();

    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const std::map<KeyFrame*, std::tuple<int, int>> observations = pMP->GetObservations();

        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                continue;
            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
                const int leftIndex = std::get<0>(mit->second);
                cv::KeyPoint kpUn;

                // Monocular left observation
                if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0) {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
                    EdgeMono* e = new EdgeMono(0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs);
                    const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                    num_edges++;
                }
                // Stereo observation
                else if (leftIndex != -1) {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                    EdgeStereo* e = new EdgeStereo(0);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));
                    const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                    num_edges++;
                }

                // Monocular right observation
                if (pKFi->mpCamera2) {
                    int rightIndex = std::get<1>(mit->second);
                    if (rightIndex != -1) {
                        rightIndex -= pKFi->NLeft;
                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;
                        EdgeMono* e = new EdgeMono(1);
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);
                        const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                        num_edges++;
                    }
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it);
    float err_end = optimizer.activeRobustChi2();
    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    std::vector<std::pair<KeyFrame*, MapPoint*>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations — Mono
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth < 10.f;
        if (pMP->isBad()) continue;
        if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(std::make_pair(pKFi, pMP));
        }
    }

    // Stereo
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if (pMP->isBad()) continue;
        if (e->chi2() > chi2Stereo2) {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(std::make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex and erase outliers
    std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

    // Convergence check
    if ((2 * err < err_end || isnan(err) || isnan(err_end)) && !bLarge) {
        std::cout << "FAIL LOCAL-INERTIAL BA!!!!" << std::endl;
        return;
    }

    if (!vToErase.empty()) {
        for (size_t i = 0; i < vToErase.size(); i++) {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    for (auto lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Recover optimized data — Local temporal Keyframes
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; i++) {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF = 0;

        if (pKFi->bImu) {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    // Local visual KeyFrame
    for (auto it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++) {
        KeyFrame* pKFi = *it;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF = 0;
    }

    // Points
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + iniMPid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
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
    params.diagonalDamping = true;
    params.lambdaInitial = 100.0;
    params.lambdaUpperBound = 1e9;
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
    params.maxIterations = 10;
    params.setVerbosity("SILENT");

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        for (KeyFrame* pKFi : sAllKFs) {
            if (pKFi->isBad() || !result.exists(X(pKFi->mnId))) continue;
            gtsam::Pose3 Twc = result.at<gtsam::Pose3>(X(pKFi->mnId));
            Sophus::SE3f Tcw = toSophus(Twc.inverse());
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
    // -------------------------------------------------------------------------
    // Full g2o-based inertial optimization (identical to G2oOptimizer).
    // Estimates gravity direction, scale, velocities, and biases from
    // fixed camera poses + IMU preintegration.
    // -------------------------------------------------------------------------
    Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
    int its = 200;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup g2o optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (priorG != 0.f)
        solver->setUserLambdaInit(1e3);
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses, optimizable velocities)
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid) continue;
        VertexPose* VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + (pKFi->mnId) + 1);
        VV->setFixed(bFixedVel);
        optimizer.addVertex(VV);
    }

    // Biases (shared across all KFs)
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid * 2 + 2);
    VG->setFixed(bFixedVel);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    VA->setFixed(bFixedVel);
    optimizer.addVertex(VA);

    // Bias priors
    Eigen::Vector3f bprior;
    bprior.setZero();
    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity direction and scale vertices
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(!bMono); // Fixed for stereo
    optimizer.addVertex(VS);

    // IMU edges with gravity and scale
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;
            if (!pKFi->mpImuPreintegrated)
                std::cout << "Not preintegrated measurement" << std::endl;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex* VG_v = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex* VA_v = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex* VGDir_v = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex* VS_v = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG_v || !VA_v || !VP2 || !VV2 || !VGDir_v || !VS_v)
                continue;

            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG_v));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA_v));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir_v));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS_v));
            optimizer.addEdge(ei);
        }
    }

    // Optimize
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover optimized data
    scale = VS->estimate();
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid * 2 + 3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();

    IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
    Rwg = VGDir->estimate().Rwg;

    // Update keyframe velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; i < N; i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid) continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01) {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        } else {
            pKFi->SetNewBias(b);
        }
    }
}

void GtsamOptimizer::InertialOptimization(Map* pMap, Eigen::Vector3d& bg,
    Eigen::Vector3d& ba, float priorG, float priorA) {
    // -------------------------------------------------------------------------
    // Bias-only inertial optimization (gravity/scale fixed).
    // Identical to G2oOptimizer version.
    // -------------------------------------------------------------------------
    int its = 200;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);
    optimizer.setAlgorithm(solver);

    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid) continue;
        VertexPose* VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + (pKFi->mnId) + 1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid * 2 + 2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Bias priors
    Eigen::Vector3f bprior;
    bprior.setZero();
    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    epa->setInformation(priorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    epg->setInformation(priorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale (fixed)
    VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(1.0);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(true);
    optimizer.addVertex(VS);

    // IMU edges
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex* VG_v = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex* VA_v = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex* VGDir_v = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex* VS_v = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG_v || !VA_v || !VP2 || !VV2 || !VGDir_v || !VS_v)
                continue;

            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG_v));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA_v));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir_v));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS_v));
            optimizer.addEdge(ei);
        }
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid * 2 + 3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

    // Update KF velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; i < N; i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid) continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01) {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        } else {
            pKFi->SetNewBias(b);
        }
    }
}

void GtsamOptimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d& Rwg,
    double& scale) {
    // -------------------------------------------------------------------------
    // Scale refinement: optimize only gravity direction + scale.
    // Uses GaussNewton with Huber kernel. Identical to G2oOptimizer.
    // -------------------------------------------------------------------------
    int its = 10;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // All KF variables fixed
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid) continue;
        VertexPose* VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + 1 + (pKFi->mnId));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Per-KF fixed biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2 * (maxKFid + 1) + (pKFi->mnId));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(3 * (maxKFid + 1) + (pKFi->mnId));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // Gravity and scale (optimizable)
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4 * (maxKFid + 1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4 * (maxKFid + 1) + 1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // IMU edges with Huber kernel
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];
        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid + 1) + pKFi->mnId);
            g2o::HyperGraph::Vertex* VG_v = optimizer.vertex(2 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VA_v = optimizer.vertex(3 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VGDir_v = optimizer.vertex(4 * (maxKFid + 1));
            g2o::HyperGraph::Vertex* VS_v = optimizer.vertex(4 * (maxKFid + 1) + 1);
            if (!VP1 || !VV1 || !VG_v || !VA_v || !VP2 || !VV2 || !VGDir_v || !VS_v)
                continue;

            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG_v));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA_v));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir_v));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS_v));
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.f);
            optimizer.addEdge(ei);
        }
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover scale and gravity
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

} // namespace ORB_SLAM3
