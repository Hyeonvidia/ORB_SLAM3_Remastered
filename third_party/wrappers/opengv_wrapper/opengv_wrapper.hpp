#pragma once

// =============================================================================
// OpenGV Wrapper — Type aliases and geometry utilities from OpenGV
// Original MLPnPsolver was modified from Steffen Urban's OpenGV fork
// =============================================================================

#include <Eigen/Dense>
#include <vector>

namespace slam3 { namespace geometry {

// ============================================================================
// Type aliases (matching OpenGV conventions)
// ============================================================================

// Bearing vectors (unit-length 3-vectors in camera frame)
using BearingVector  = Eigen::Vector3d;
using BearingVectors = std::vector<BearingVector, Eigen::aligned_allocator<BearingVector>>;

// Covariance matrices
using Covariance2d  = Eigen::Matrix2d;
using Covariance3d  = Eigen::Matrix3d;
using Covariances3d = std::vector<Covariance3d, Eigen::aligned_allocator<Covariance3d>>;

// 3D points
using Point3d  = Eigen::Vector3d;
using Points3d = std::vector<Point3d, Eigen::aligned_allocator<Point3d>>;

// Homogeneous 3D points
using Point4d  = Eigen::Vector4d;
using Points4d = std::vector<Point4d, Eigen::aligned_allocator<Point4d>>;

// Rotation and transformation
using Rodrigues      = Eigen::Vector3d;
using Rotation       = Eigen::Matrix3d;
using Transformation = Eigen::Matrix<double, 3, 4>;
using Translation    = Eigen::Vector3d;

// ============================================================================
// Geometry utilities
// ============================================================================

// Rodrigues vector to rotation matrix
Rotation rodrigues2rot(const Eigen::Vector3d &omega);

// Rotation matrix to Rodrigues vector
Rodrigues rot2rodrigues(const Rotation &R);

// Pixel coordinates to normalized bearing vector
BearingVector pixelToBearing(double u, double v,
                             double fx, double fy, double cx, double cy);

}} // namespace slam3::geometry
