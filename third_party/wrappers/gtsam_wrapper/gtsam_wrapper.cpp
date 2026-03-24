// =============================================================================
// ORB_SLAM3_Remastered — GTSAM Wrapper Implementation
// =============================================================================
#include "gtsam_wrapper.hpp"
#include <cmath>

namespace ORB_SLAM3 {
namespace gtsam_wrapper {

// ---------------------------------------------------------------------------
// Right Jacobian of SO(3)
// Jr(v) = I - (1-cos(theta))/theta^2 * [v]x + (theta - sin(theta))/theta^3 * [v]x^2
// ---------------------------------------------------------------------------

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d& v) {
    const double theta2 = v.squaredNorm();
    const double theta = std::sqrt(theta2);
    const Eigen::Matrix3d W = Skew(v);

    if (theta < 1e-5) {
        return Eigen::Matrix3d::Identity() - 0.5 * W;
    }

    return Eigen::Matrix3d::Identity()
           - (1.0 - std::cos(theta)) / theta2 * W
           + (theta - std::sin(theta)) / (theta2 * theta) * W * W;
}

Eigen::Matrix3d RightJacobianSO3(double x, double y, double z) {
    return RightJacobianSO3(Eigen::Vector3d(x, y, z));
}

// ---------------------------------------------------------------------------
// Inverse Right Jacobian of SO(3)
// Jr^{-1}(v) = I + 0.5*[v]x + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta))) * [v]x^2
// ---------------------------------------------------------------------------

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v) {
    const double theta2 = v.squaredNorm();
    const double theta = std::sqrt(theta2);
    const Eigen::Matrix3d W = Skew(v);

    if (theta < 1e-5) {
        return Eigen::Matrix3d::Identity() + 0.5 * W;
    }

    return Eigen::Matrix3d::Identity()
           + 0.5 * W
           + (1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta))) * W * W;
}

Eigen::Matrix3d InverseRightJacobianSO3(double x, double y, double z) {
    return InverseRightJacobianSO3(Eigen::Vector3d(x, y, z));
}

} // namespace gtsam_wrapper
} // namespace ORB_SLAM3
