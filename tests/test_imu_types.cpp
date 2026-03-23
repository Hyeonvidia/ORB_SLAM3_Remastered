/**
 * Tier 1 GTest: IMU data types
 */

#include <gtest/gtest.h>
#include "ImuTypes.hpp"

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <cmath>

// ---------------------------------------------------------------------------
// IMU::Point constructor: verify a, w, t values
// ---------------------------------------------------------------------------
TEST(ImuTypes, PointConstructor)
{
    ORB_SLAM3::IMU::Point pt(1.0f, 2.0f, 3.0f,   // acc
                             0.1f, 0.2f, 0.3f,    // gyro
                             12345.0);             // timestamp

    EXPECT_NEAR(pt.a(0), 1.0f, 1e-6f);
    EXPECT_NEAR(pt.a(1), 2.0f, 1e-6f);
    EXPECT_NEAR(pt.a(2), 3.0f, 1e-6f);

    EXPECT_NEAR(pt.w(0), 0.1f, 1e-6f);
    EXPECT_NEAR(pt.w(1), 0.2f, 1e-6f);
    EXPECT_NEAR(pt.w(2), 0.3f, 1e-6f);

    EXPECT_DOUBLE_EQ(pt.t, 12345.0);
}

// ---------------------------------------------------------------------------
// IMU::Bias default: all zeros
// ---------------------------------------------------------------------------
TEST(ImuTypes, BiasDefault)
{
    ORB_SLAM3::IMU::Bias b;

    EXPECT_FLOAT_EQ(b.bax, 0.0f);
    EXPECT_FLOAT_EQ(b.bay, 0.0f);
    EXPECT_FLOAT_EQ(b.baz, 0.0f);
    EXPECT_FLOAT_EQ(b.bwx, 0.0f);
    EXPECT_FLOAT_EQ(b.bwy, 0.0f);
    EXPECT_FLOAT_EQ(b.bwz, 0.0f);
}

// ---------------------------------------------------------------------------
// IMU::Bias parameterized + CopyFrom
// ---------------------------------------------------------------------------
TEST(ImuTypes, BiasParameterizedAndCopy)
{
    ORB_SLAM3::IMU::Bias b1(0.1f, 0.2f, 0.3f, 0.01f, 0.02f, 0.03f);

    EXPECT_FLOAT_EQ(b1.bax, 0.1f);
    EXPECT_FLOAT_EQ(b1.bay, 0.2f);
    EXPECT_FLOAT_EQ(b1.baz, 0.3f);
    EXPECT_FLOAT_EQ(b1.bwx, 0.01f);
    EXPECT_FLOAT_EQ(b1.bwy, 0.02f);
    EXPECT_FLOAT_EQ(b1.bwz, 0.03f);

    ORB_SLAM3::IMU::Bias b2;
    b2.CopyFrom(b1);

    EXPECT_FLOAT_EQ(b2.bax, b1.bax);
    EXPECT_FLOAT_EQ(b2.bay, b1.bay);
    EXPECT_FLOAT_EQ(b2.baz, b1.baz);
    EXPECT_FLOAT_EQ(b2.bwx, b1.bwx);
    EXPECT_FLOAT_EQ(b2.bwy, b1.bwy);
    EXPECT_FLOAT_EQ(b2.bwz, b1.bwz);
}

// ---------------------------------------------------------------------------
// IMU::Calib constructor: verify mbIsSet, Tbc/Tcb inverse
// ---------------------------------------------------------------------------
TEST(ImuTypes, CalibConstructor)
{
    Sophus::SE3f Tbc; // identity
    float ng  = 1e-3f;
    float na  = 1e-3f;
    float ngw = 1e-4f;
    float naw = 1e-4f;

    ORB_SLAM3::IMU::Calib calib(Tbc, ng, na, ngw, naw);

    EXPECT_TRUE(calib.mbIsSet);

    // Tbc * Tcb should be identity
    Sophus::SE3f product = calib.mTbc * calib.mTcb;
    Eigen::Matrix4f prodMat = product.matrix();
    Eigen::Matrix4f eye4 = Eigen::Matrix4f::Identity();

    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_NEAR(prodMat(r, c), eye4(r, c), 1e-5f);
}

// ---------------------------------------------------------------------------
// IMU::Calib default: mbIsSet should be false
// ---------------------------------------------------------------------------
TEST(ImuTypes, CalibDefaultNotSet)
{
    ORB_SLAM3::IMU::Calib calib;
    EXPECT_FALSE(calib.mbIsSet);
}

// ---------------------------------------------------------------------------
// IMU::IntegratedRotation: zero angular velocity -> identity deltaR
// ---------------------------------------------------------------------------
TEST(ImuTypes, IntegratedRotation_ZeroAngVel)
{
    Eigen::Vector3f zeroAngVel = Eigen::Vector3f::Zero();
    ORB_SLAM3::IMU::Bias zeroBias;
    float dt = 0.01f;

    ORB_SLAM3::IMU::IntegratedRotation ir(zeroAngVel, zeroBias, dt);

    Eigen::Matrix3f eye3 = Eigen::Matrix3f::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(ir.deltaR(r, c), eye3(r, c), 1e-5f);
}

// ---------------------------------------------------------------------------
// IMU::Preintegrated: default constructor state
// ---------------------------------------------------------------------------
TEST(ImuTypes, Preintegrated_DefaultState)
{
    ORB_SLAM3::IMU::Preintegrated preint;
    // Default-constructed Preintegrated has no measurements accumulated
    // dT should be zero (uninitialized default)
    // This test verifies the object can be constructed without crashing
    SUCCEED();
}

// ---------------------------------------------------------------------------
// IMU::Preintegrated: IntegrateNewMeasurement accumulation
// ---------------------------------------------------------------------------
TEST(ImuTypes, Preintegrated_Integrate)
{
    Sophus::SE3f Tbc; // identity
    float ng  = 1e-3f;
    float na  = 1e-3f;
    float ngw = 1e-5f;
    float naw = 1e-5f;

    ORB_SLAM3::IMU::Calib calib(Tbc, ng, na, ngw, naw);
    ORB_SLAM3::IMU::Bias zeroBias;

    ORB_SLAM3::IMU::Preintegrated preint(zeroBias, calib);

    EXPECT_FLOAT_EQ(preint.dT, 0.0f);

    Eigen::Vector3f acc(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro = Eigen::Vector3f::Zero();
    float dt = 0.01f;

    // Integrate 10 measurements
    for (int i = 0; i < 10; ++i)
        preint.IntegrateNewMeasurement(acc, gyro, dt);

    // Total integrated time should be 10 * 0.01 = 0.1
    EXPECT_NEAR(preint.dT, 0.1f, 1e-5f);

    // dP should be non-zero after integration with acceleration
    float dpNorm = preint.dP.norm();
    EXPECT_GT(dpNorm, 0.0f);

    // dV should be non-zero
    float dvNorm = preint.dV.norm();
    EXPECT_GT(dvNorm, 0.0f);
}

// ---------------------------------------------------------------------------
// RightJacobianSO3: zero vector -> identity matrix
// ---------------------------------------------------------------------------
TEST(ImuTypes, RightJacobianSO3_Zero)
{
    Eigen::Vector3f zero = Eigen::Vector3f::Zero();
    Eigen::Matrix3f J = ORB_SLAM3::IMU::RightJacobianSO3(zero);

    Eigen::Matrix3f eye3 = Eigen::Matrix3f::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(J(r, c), eye3(r, c), 1e-5f);
}

// ---------------------------------------------------------------------------
// NormalizeRotation: identity stays identity
// ---------------------------------------------------------------------------
TEST(ImuTypes, NormalizeRotation_Identity)
{
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f result = ORB_SLAM3::IMU::NormalizeRotation(I);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(result(r, c), I(r, c), 1e-5f);
}
