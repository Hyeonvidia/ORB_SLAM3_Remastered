#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3 {

// Backend-agnostic Sim3 representation.
// Stores rotation, translation, and scale as Eigen types.
// Convertible to/from g2o::Sim3 and gtsam::Similarity3 via Converter.
struct Sim3Type {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double s;

    Sim3Type()
        : R(Eigen::Matrix3d::Identity()), t(Eigen::Vector3d::Zero()), s(1.0) {}

    Sim3Type(const Eigen::Matrix3d& R_, const Eigen::Vector3d& t_, double s_)
        : R(R_), t(t_), s(s_) {}

    Sim3Type(const Eigen::Quaterniond& q, const Eigen::Vector3d& t_, double s_)
        : R(q.toRotationMatrix()), t(t_), s(s_) {}

    Sim3Type inverse() const;
    Eigen::Vector3d map(const Eigen::Vector3d& p) const { return s * R * p + t; }
    Sim3Type operator*(const Sim3Type& other) const;

    Eigen::Quaterniond rotation() const { return Eigen::Quaterniond(R); }
    Eigen::Vector3d translation() const { return t; }
    double scale() const { return s; }
};

} // namespace ORB_SLAM3
