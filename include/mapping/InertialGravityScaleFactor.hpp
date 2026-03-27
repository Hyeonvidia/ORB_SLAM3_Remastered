/**
 * GTSAM-native replacement for g2o EdgeInertialGS.
 *
 * Inertial factor that estimates gravity direction and scale from
 * fixed camera poses + IMU preintegration.
 *
 * Three use cases (determined by which variables are fixed via priors):
 *   1. Full init:       V1, V2, bg, ba, Rwg, scale all free
 *   2. Bias-only:       Rwg, scale fixed via Constrained priors
 *   3. Scale refine:    V1, V2, bg, ba fixed via Constrained priors
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <Eigen/Core>
#include "ImuTypes.hpp"

namespace ORB_SLAM3 {

/**
 * 6-key factor: V1, V2, GyroBias, AccBias, GravityRot(Rot3), Scale(Vector1)
 *
 * Poses (Rwb1, twb1, Rwb2, twb2) are always FIXED and passed as constructor
 * parameters — they never participate in optimization.
 *
 * Error (9D) = [er(3), ev(3), ep(3)]:
 *   er = LogSO3(dR(bg)^T * Rwb1^T * Rwb2)
 *   ev = Rwb1^T * (s*(v2 - v1) - g*dt) - dV(bg,ba)
 *   ep = Rwb1^T * (s*(twb2 - twb1 - v1*dt) - g*dt^2/2) - dP(bg,ba)
 *
 * where g = Rwg * [0, 0, -GRAVITY_VALUE]^T
 */
class InertialGSFactor
    : public gtsam::NoiseModelFactorN<
          gtsam::Vector3,  // V1
          gtsam::Vector3,  // V2
          gtsam::Vector3,  // gyro bias
          gtsam::Vector3,  // acc bias
          gtsam::Rot3,     // Rwg (gravity direction)
          gtsam::Vector1   // scale
      > {
 public:
    using Base = gtsam::NoiseModelFactorN<
        gtsam::Vector3, gtsam::Vector3,
        gtsam::Vector3, gtsam::Vector3,
        gtsam::Rot3, gtsam::Vector1>;

    InertialGSFactor(
        gtsam::Key v1Key, gtsam::Key v2Key,
        gtsam::Key bgKey, gtsam::Key baKey,
        gtsam::Key rwgKey, gtsam::Key scaleKey,
        const Eigen::Matrix3d& Rwb1, const Eigen::Vector3d& twb1,
        const Eigen::Matrix3d& Rwb2, const Eigen::Vector3d& twb2,
        IMU::Preintegrated* pInt,
        const gtsam::SharedNoiseModel& model)
        : Base(model, v1Key, v2Key, bgKey, baKey, rwgKey, scaleKey),
          Rwb1_(Rwb1), twb1_(twb1), Rwb2_(Rwb2), twb2_(twb2),
          pInt_(pInt), dt_(pInt->dT) {
        gI_ << 0, 0, -IMU::GRAVITY_VALUE;
    }

    gtsam::Vector evaluateError(
        const gtsam::Vector3& v1,
        const gtsam::Vector3& v2,
        const gtsam::Vector3& bg,
        const gtsam::Vector3& ba,
        const gtsam::Rot3& Rwg,
        const gtsam::Vector1& logScale,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3,
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5,
        gtsam::OptionalMatrixType H6) const override {
        // ---- Compute preintegrated deltas with current bias ----
        const IMU::Bias bias(ba[0], ba[1], ba[2], bg[0], bg[1], bg[2]);
        const Eigen::Matrix3d dR = pInt_->GetDeltaRotation(bias).cast<double>();
        const Eigen::Vector3d dV = pInt_->GetDeltaVelocity(bias).cast<double>();
        const Eigen::Vector3d dP = pInt_->GetDeltaPosition(bias).cast<double>();

        // ---- Gravity and scale ----
        const Eigen::Vector3d g = Rwg.matrix() * gI_;
        const double s = logScale(0);

        // ---- Body frame quantities ----
        const Eigen::Matrix3d Rbw1 = Rwb1_.transpose();

        // ---- 9D Error ----
        const Eigen::Vector3d er = LogSO3(dR.transpose() * Rbw1 * Rwb2_);
        const Eigen::Vector3d ev = Rbw1 * (s * (v2 - v1) - g * dt_) - dV;
        const Eigen::Vector3d ep = Rbw1 * (s * (twb2_ - twb1_ - v1 * dt_) - g * dt_ * dt_ / 2.0) - dP;

        gtsam::Vector9 error;
        error << er, ev, ep;

        // ---- Analytical Jacobians ----
        if (H1 || H2 || H3 || H4 || H5 || H6) {
            const IMU::Bias db = pInt_->GetDeltaBias(bias);
            Eigen::Vector3d dbg;
            dbg << db.bwx, db.bwy, db.bwz;

            const Eigen::Matrix3d eR = dR.transpose() * Rbw1 * Rwb2_;
            const Eigen::Vector3d er_val = LogSO3(eR);
            const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er_val);

            // Preintegration Jacobians
            const Eigen::Matrix3d JRg = pInt_->JRg.cast<double>();
            const Eigen::Matrix3d JVg = pInt_->JVg.cast<double>();
            const Eigen::Matrix3d JPg = pInt_->JPg.cast<double>();
            const Eigen::Matrix3d JVa = pInt_->JVa.cast<double>();
            const Eigen::Matrix3d JPa = pInt_->JPa.cast<double>();

            // H1: Jacobian wrt V1 (3x3 in rows 3-8)
            if (H1) {
                H1->resize(9, 3);
                H1->setZero();
                H1->block<3,3>(3,0) = -s * Rbw1;         // dEv/dv1
                H1->block<3,3>(6,0) = -s * Rbw1 * dt_;    // dEp/dv1
            }

            // H2: Jacobian wrt V2 (3x3 in row 3-5)
            if (H2) {
                H2->resize(9, 3);
                H2->setZero();
                H2->block<3,3>(3,0) = s * Rbw1;           // dEv/dv2
            }

            // H3: Jacobian wrt GyroBias (3x3)
            if (H3) {
                H3->resize(9, 3);
                H3->setZero();
                H3->block<3,3>(0,0) = -invJr * eR.transpose() * RightJacobianSO3(JRg * dbg) * JRg;
                H3->block<3,3>(3,0) = -JVg;
                H3->block<3,3>(6,0) = -JPg;
            }

            // H4: Jacobian wrt AccBias (3x3)
            if (H4) {
                H4->resize(9, 3);
                H4->setZero();
                H4->block<3,3>(3,0) = -JVa;
                H4->block<3,3>(6,0) = -JPa;
            }

            // H5: Jacobian wrt Rwg (Rot3, 3-DOF)
            if (H5) {
                // g = Rwg * gI  →  dg/dRwg = -Rwg * [gI]x  (3x3, right perturbation)
                // But GTSAM Rot3 uses 3-DOF. For the 2-DOF gravity from g2o,
                // the z-rotation produces zero gradient naturally.
                Eigen::MatrixXd dGdTheta = -Rwg.matrix() * gtsam::skewSymmetric(gI_);
                H5->resize(9, 3);
                H5->setZero();
                H5->block<3,3>(3,0) = -Rbw1 * dGdTheta * dt_;
                H5->block<3,3>(6,0) = -0.5 * Rbw1 * dGdTheta * dt_ * dt_;
            }

            // H6: Jacobian wrt scale (1-DOF)
            if (H6) {
                H6->resize(9, 1);
                H6->setZero();
                H6->block<3,1>(3,0) = Rbw1 * (v2 - v1);
                H6->block<3,1>(6,0) = Rbw1 * (twb2_ - twb1_ - v1 * dt_);
            }
        }

        return error;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<InertialGSFactor>(*this);
    }

 private:
    Eigen::Matrix3d Rwb1_, Rwb2_;
    Eigen::Vector3d twb1_, twb2_;
    IMU::Preintegrated* pInt_;
    double dt_;
    Eigen::Vector3d gI_;

    // ---- SO3 helpers (same as G2oTypes.cpp) ----
    static Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
        const double d2 = w.squaredNorm();
        const double d = std::sqrt(d2);
        Eigen::Matrix3d W = gtsam::skewSymmetric(w);
        if (d < 1e-5)
            return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
        return Eigen::Matrix3d::Identity() + W * std::sin(d) / d + W * W * (1.0 - std::cos(d)) / d2;
    }

    static Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
        const double tr = R.trace();
        Eigen::Vector3d w;
        w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
        const double costheta = (tr - 1.0) * 0.5;
        if (costheta > 1 || costheta < -1) return w;
        const double theta = std::acos(costheta);
        const double s = std::sin(theta);
        if (std::fabs(s) < 1e-5) return w;
        return theta * w / s;
    }

    static Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d& v) {
        const double d2 = v.squaredNorm();
        const double d = std::sqrt(d2);
        Eigen::Matrix3d W = gtsam::skewSymmetric(v);
        if (d < 1e-5) return Eigen::Matrix3d::Identity();
        return Eigen::Matrix3d::Identity() - W*(1.0-std::cos(d))/d2 + W*W*(d-std::sin(d))/(d2*d);
    }

    static Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v) {
        const double d2 = v.squaredNorm();
        const double d = std::sqrt(d2);
        Eigen::Matrix3d W = gtsam::skewSymmetric(v);
        if (d < 1e-5) return Eigen::Matrix3d::Identity();
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+std::cos(d))/(2.0*d*std::sin(d)));
    }
};

}  // namespace ORB_SLAM3
