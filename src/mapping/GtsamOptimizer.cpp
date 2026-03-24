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
#include <gtsam/slam/PriorFactor.h>

#include <boost/optional.hpp>
#include <mutex>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ORB_SLAM3
{

using gtsam::symbol_shorthand::X;  // Pose keys: X(i)
using gtsam::symbol_shorthand::L;  // Landmark keys: L(i)

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

void GtsamOptimizer::FullInertialBA(Map* /*pMap*/, int /*its*/, bool /*bFixLocal*/,
    unsigned long /*nLoopKF*/, bool* /*pbStopFlag*/, bool /*bInit*/,
    float /*priorG*/, float /*priorA*/, Eigen::VectorXd* /*vSingVal*/, bool* /*bHess*/) {
    throw std::runtime_error("GtsamOptimizer::FullInertialBA not yet implemented");
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

    // Add MPs and projection factors
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

            if (pKFi->mvuRight[leftIndex] < 0) {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Vector2d ob; ob << kpUn.pt.x, kpUn.pt.y;
                float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise2(invSigma2, thHuber2D);
                graph.emplace_shared<MonoProjectionFactor>(
                    X(pKFi->mnId), lmKey, ob, pKFi->mpCamera, noise);
            } else {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Vector3d ob;
                ob << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                auto noise = makeHuberNoise3(invSigma2, thHuber3D);
                graph.emplace_shared<StereoProjectionFactor>(
                    X(pKFi->mnId), lmKey, ob,
                    pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy, pKFi->mbf, noise);
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

    // Guard: discard catastrophic LBA results
    if (avgMPShift > 0.5) {
        std::cerr << "[LBA-GUARD] DISCARDED avgPoseShift=" << avgPoseShift
                  << " avgMPShift=" << avgMPShift << " (>0.5m threshold)" << std::endl;
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
                      << " nKF=" << nPoseRecovered << " nMP=" << nMPRecovered << std::endl;
        }
    }

    pMap->IncreaseChangeIndex();
}

void GtsamOptimizer::LocalBundleAdjustment(KeyFrame* pMainKF,
    std::vector<KeyFrame*> vpAdjustKF, std::vector<KeyFrame*> vpFixedKF,
    bool* pbStopFlag) {
    // Welding-area LocalBA stub -- rarely called, not yet ported
    std::cerr << "[GtsamOptimizer::LocalBundleAdjustment(welding)] stub -- not yet implemented" << std::endl;
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
