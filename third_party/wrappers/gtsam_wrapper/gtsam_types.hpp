// =============================================================================
// ORB_SLAM3_Remastered — GTSAM Types
// Type definitions replacing original g2o custom types (G2oTypes.h)
// Provides Eigen-based vector/matrix aliases used throughout the SLAM system
// =============================================================================
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

namespace ORB_SLAM3 {
namespace types {

// ---------------------------------------------------------------------------
// Vector/Matrix type aliases (from original G2oTypes.h)
// ---------------------------------------------------------------------------
using Vector6d  = Eigen::Matrix<double, 6, 1>;
using Vector9d  = Eigen::Matrix<double, 9, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Vector15d = Eigen::Matrix<double, 15, 1>;

using Matrix9d  = Eigen::Matrix<double, 9, 9>;
using Matrix12d = Eigen::Matrix<double, 12, 12>;
using Matrix15d = Eigen::Matrix<double, 15, 15>;

} // namespace types
} // namespace ORB_SLAM3
