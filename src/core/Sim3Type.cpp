#include "core/Sim3Type.hpp"

namespace ORB_SLAM3 {

Sim3Type Sim3Type::inverse() const {
    Eigen::Matrix3d Rinv = R.transpose();
    double sinv = 1.0 / s;
    Eigen::Vector3d tinv = -sinv * Rinv * t;
    return Sim3Type(Rinv, tinv, sinv);
}

Sim3Type Sim3Type::operator*(const Sim3Type& other) const {
    Eigen::Matrix3d R3 = R * other.R;
    Eigen::Vector3d t3 = s * R * other.t + t;
    double s3 = s * other.s;
    return Sim3Type(R3, t3, s3);
}

} // namespace ORB_SLAM3
