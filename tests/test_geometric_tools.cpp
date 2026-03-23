/**
 * Tier 1 GTest: Geometric computation (triangulation)
 */

#include <gtest/gtest.h>
#include "GeometricTools.hpp"

#include <Eigen/Dense>
#include <cmath>

// ---------------------------------------------------------------------------
// Triangulate: two cameras looking at a known 3D point
// Camera 1 at origin, Camera 2 translated along X by 1.0
// Point at (0.5, 0.0, 2.0) in world
// ---------------------------------------------------------------------------
TEST(GeometricTools, Triangulate_KnownPoint)
{
    // Known 3D point in world coordinates
    Eigen::Vector3f pWorld(0.5f, 0.0f, 2.0f);

    // Camera 1: identity pose (R=I, t=0) -> Tc1w = [I | 0]
    Eigen::Matrix<float, 3, 4> Tc1w;
    Tc1w.setZero();
    Tc1w.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

    // Camera 2: translated 1.0 along X -> Tc2w = [I | -1, 0, 0]
    // World to cam2: t_c2 = R * pw + t => with R=I, t=(-1,0,0)
    Eigen::Matrix<float, 3, 4> Tc2w;
    Tc2w.setZero();
    Tc2w.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
    Tc2w(0, 3) = -1.0f;

    // Project pWorld into each camera (normalized coords)
    Eigen::Vector3f pc1 = Tc1w.block<3, 3>(0, 0) * pWorld + Tc1w.col(3);
    Eigen::Vector3f pc2 = Tc2w.block<3, 3>(0, 0) * pWorld + Tc2w.col(3);

    // Normalized bearing vectors
    Eigen::Vector3f x_c1 = pc1 / pc1(2);
    Eigen::Vector3f x_c2 = pc2 / pc2(2);

    Eigen::Vector3f x3D;
    bool success = ORB_SLAM3::GeometricTools::Triangulate(x_c1, x_c2, Tc1w, Tc2w, x3D);

    EXPECT_TRUE(success);
    EXPECT_NEAR(x3D(0), pWorld(0), 1e-3f);
    EXPECT_NEAR(x3D(1), pWorld(1), 1e-3f);
    EXPECT_NEAR(x3D(2), pWorld(2), 1e-3f);
}

// ---------------------------------------------------------------------------
// Triangulate: parallel rays -> should fail or produce degenerate point
// Both cameras at the same position looking at the same bearing direction
// ---------------------------------------------------------------------------
TEST(GeometricTools, Triangulate_ParallelRays)
{
    // Both cameras at origin (zero baseline)
    Eigen::Matrix<float, 3, 4> Tc1w;
    Tc1w.setZero();
    Tc1w.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

    Eigen::Matrix<float, 3, 4> Tc2w;
    Tc2w.setZero();
    Tc2w.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

    // Same bearing in both cameras (parallel rays from same point)
    Eigen::Vector3f x_c1(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f x_c2(0.0f, 0.0f, 1.0f);

    Eigen::Vector3f x3D;
    bool success = ORB_SLAM3::GeometricTools::Triangulate(x_c1, x_c2, Tc1w, Tc2w, x3D);

    // With zero baseline, triangulation should either fail (return false)
    // or produce a degenerate result (very large / inf / nan coordinates)
    if (success)
    {
        // If it claims success, the point should be degenerate
        bool isDegenerate = !std::isfinite(x3D(0)) || !std::isfinite(x3D(1)) ||
                            !std::isfinite(x3D(2)) || x3D.norm() > 1e6f ||
                            x3D.norm() < 1e-6f;
        EXPECT_TRUE(isDegenerate)
            << "Parallel rays should produce degenerate triangulation. "
            << "Got: (" << x3D(0) << ", " << x3D(1) << ", " << x3D(2) << ")";
    }
    else
    {
        SUCCEED(); // Returning false is the correct behavior
    }
}
