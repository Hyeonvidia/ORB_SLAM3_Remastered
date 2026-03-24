// =============================================================================
// ORB_SLAM3_Remastered — GTSAM Wrapper
// Replaces g2o + Sophus with GTSAM equivalents
//
// Mapping:
//   Sophus::SE3f        → gtsam::Pose3 (via SE3f alias + converters)
//   g2o::SE3Quat        → gtsam::Pose3
//   g2o::Sim3            → gtsam::Similarity3
//   g2o::VertexSBAPointXYZ → gtsam::Point3 (in factor graph)
//   ExpSO3/LogSO3       → gtsam::Rot3::Expmap/Logmap
//   RightJacobianSO3    → gtsam::SO3::ExpmapDerivative
//   EdgeMono/Stereo      → gtsam::GenericProjectionFactor
//   EdgeInertial          → gtsam::CombinedImuFactor / ImuFactor
//   EdgeGyroRW/AccRW      → gtsam::BetweenFactor<Vector3>
// =============================================================================
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include "gtsam_types.hpp"

namespace ORB_SLAM3 {
namespace gtsam_wrapper {

// ===========================================================================
// Lie Group Utilities (replaces Sophus + G2oTypes.h free functions)
// ===========================================================================

/// Exponential map SO(3): Lie algebra so3 → SO(3)
/// Replaces: ExpSO3(const Eigen::Vector3d& w) from G2oTypes.h
inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
    return gtsam::Rot3::Expmap(w).matrix();
}

/// Exponential map SO(3) from individual components
inline Eigen::Matrix3d ExpSO3(double x, double y, double z) {
    return ExpSO3(Eigen::Vector3d(x, y, z));
}

/// Logarithmic map SO(3): SO(3) → so3
/// Replaces: LogSO3(const Eigen::Matrix3d& R) from G2oTypes.h
inline Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
    return gtsam::Rot3::Logmap(gtsam::Rot3(R));
}

/// Skew-symmetric matrix from 3-vector
/// Replaces: Skew(const Eigen::Vector3d& w) from G2oTypes.h
inline Eigen::Matrix3d Skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d W;
    W <<     0, -w(2),  w(1),
          w(2),     0, -w(0),
         -w(1),  w(0),     0;
    return W;
}

/// Right Jacobian of SO(3)
/// Replaces: RightJacobianSO3 from G2oTypes.h
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d& v);
Eigen::Matrix3d RightJacobianSO3(double x, double y, double z);

/// Inverse Right Jacobian of SO(3)
/// Replaces: InverseRightJacobianSO3 from G2oTypes.h
Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v);
Eigen::Matrix3d InverseRightJacobianSO3(double x, double y, double z);

/// Normalize rotation matrix via SVD
/// Replaces: NormalizeRotation from G2oTypes.h
template<typename T = double>
Eigen::Matrix<T, 3, 3> NormalizeRotation(const Eigen::Matrix<T, 3, 3>& R) {
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

// ===========================================================================
// SE3 Pose Type (replaces Sophus::SE3f / Sophus::SE3d)
// ===========================================================================

/// SE3 pose wrapper using gtsam::Pose3
/// Provides interface compatible with original Sophus::SE3f usage
class SE3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3() : pose_() {}
    explicit SE3(const gtsam::Pose3& pose) : pose_(pose) {}
    SE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : pose_(gtsam::Rot3(R), gtsam::Point3(t)) {}
    SE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
        : pose_(gtsam::Rot3(q), gtsam::Point3(t)) {}

    /// Construct from 4x4 homogeneous matrix
    static SE3 fromMatrix(const Eigen::Matrix4d& T) {
        return SE3(gtsam::Pose3(T));
    }

    /// Rotation matrix (3x3)
    Eigen::Matrix3d rotationMatrix() const { return pose_.rotation().matrix(); }
    Eigen::Matrix3f rotationMatrixf() const { return rotationMatrix().cast<float>(); }

    /// Translation vector
    Eigen::Vector3d translation() const { return pose_.translation(); }
    Eigen::Vector3f translationf() const { return translation().cast<float>(); }

    /// Quaternion
    Eigen::Quaterniond quaternion() const { return pose_.rotation().toQuaternion(); }

    /// 4x4 homogeneous matrix
    Eigen::Matrix4d matrix() const { return pose_.matrix(); }
    Eigen::Matrix4f matrixf() const { return matrix().cast<float>(); }

    /// Inverse pose
    SE3 inverse() const { return SE3(pose_.inverse()); }

    /// Composition
    SE3 operator*(const SE3& other) const { return SE3(pose_ * other.pose_); }

    /// Transform a 3D point
    Eigen::Vector3d map(const Eigen::Vector3d& point) const {
        return pose_.transformFrom(point);
    }

    /// Log map: SE3 → se3 (6-vector)
    Eigen::Matrix<double, 6, 1> log() const {
        return gtsam::Pose3::Logmap(pose_);
    }

    /// Exp map: se3 → SE3
    static SE3 exp(const Eigen::Matrix<double, 6, 1>& xi) {
        return SE3(gtsam::Pose3::Expmap(xi));
    }

    /// Access underlying gtsam::Pose3
    const gtsam::Pose3& pose() const { return pose_; }
    gtsam::Pose3& pose() { return pose_; }

private:
    gtsam::Pose3 pose_;
};

// Float version alias (replacing Sophus::SE3f)
using SE3f = SE3;
using SE3d = SE3;

// ===========================================================================
// Sim3 Type (replaces g2o::Sim3)
// ===========================================================================

/// Similarity3 wrapper using gtsam::Similarity3
class Sim3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sim3() : sim_() {}
    explicit Sim3(const gtsam::Similarity3& sim) : sim_(sim) {}
    Sim3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, double s)
        : sim_(gtsam::Rot3(R), gtsam::Point3(t), s) {}

    Eigen::Matrix3d rotation() const { return sim_.rotation().matrix(); }
    Eigen::Vector3d translation() const { return sim_.translation(); }
    double scale() const { return sim_.scale(); }

    /// Transform a 3D point: s*R*p + t
    Eigen::Vector3d map(const Eigen::Vector3d& p) const {
        return scale() * rotation() * p + translation();
    }

    Sim3 inverse() const { return Sim3(sim_.inverse()); }
    Sim3 operator*(const Sim3& other) const {
        // Manual composition: s3 = s1*s2, R3 = R1*R2, t3 = s1*R1*t2 + t1
        double s = scale() * other.scale();
        Eigen::Matrix3d R = rotation() * other.rotation();
        Eigen::Vector3d t = scale() * rotation() * other.translation() + translation();
        return Sim3(R, t, s);
    }

    const gtsam::Similarity3& sim() const { return sim_; }

private:
    gtsam::Similarity3 sim_;
};

// ===========================================================================
// Conversion utilities (replaces Converter.h g2o functions)
// ===========================================================================

/// Convert gtsam::Pose3 to Eigen 4x4 matrix
inline Eigen::Matrix4f toMatrix4f(const gtsam::Pose3& pose) {
    return pose.matrix().cast<float>();
}

/// Convert SE3 to Eigen::Isometry3d
inline Eigen::Isometry3d toIsometry3d(const SE3& se3) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = se3.rotationMatrix();
    T.translation() = se3.translation();
    return T;
}

/// Convert Eigen 4x4 matrix to gtsam::Pose3
inline gtsam::Pose3 toPose3(const Eigen::Matrix4d& T) {
    return gtsam::Pose3(T);
}

inline gtsam::Pose3 toPose3(const Eigen::Matrix4f& T) {
    return gtsam::Pose3(T.cast<double>());
}

// ===========================================================================
// IMU Bias wrapper (replaces original IMU::Bias interaction with g2o)
// ===========================================================================

/// IMU bias combining accelerometer and gyroscope biases
/// Compatible with gtsam::imuBias::ConstantBias
inline gtsam::imuBias::ConstantBias toGtsamBias(
    const Eigen::Vector3d& ba, const Eigen::Vector3d& bg) {
    return gtsam::imuBias::ConstantBias(ba, bg);
}

} // namespace gtsam_wrapper
} // namespace ORB_SLAM3
