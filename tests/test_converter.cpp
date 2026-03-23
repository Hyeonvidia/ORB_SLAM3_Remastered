/**
 * Tier 1 GTest: Converter static conversion functions
 */

#include <gtest/gtest.h>
#include "Converter.hpp"
#include "ImuTypes.hpp"

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <cmath>

// ---------------------------------------------------------------------------
// toDescriptorVector: 5x32 Mat -> vector of 5 single-row Mats
// ---------------------------------------------------------------------------
TEST(Converter, toDescriptorVector)
{
    cv::Mat descriptors(5, 32, CV_8UC1);
    cv::randu(descriptors, cv::Scalar(0), cv::Scalar(255));

    std::vector<cv::Mat> vDesc = ORB_SLAM3::Converter::toDescriptorVector(descriptors);

    ASSERT_EQ(vDesc.size(), static_cast<std::size_t>(5));
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(vDesc[i].rows, 1);
        EXPECT_EQ(vDesc[i].cols, 32);
        // Verify contents match the original row
        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(vDesc[i].at<unsigned char>(0, j),
                       descriptors.at<unsigned char>(i, j));
        }
    }
}

// ---------------------------------------------------------------------------
// toCvMat(Eigen::Matrix3f) roundtrip with toMatrix3f
// ---------------------------------------------------------------------------
TEST(Converter, Matrix3f_Roundtrip)
{
    Eigen::Matrix3f eigIn;
    eigIn << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f,
             7.0f, 8.0f, 9.0f;

    cv::Mat cvMat = ORB_SLAM3::Converter::toCvMat(eigIn);
    ASSERT_EQ(cvMat.rows, 3);
    ASSERT_EQ(cvMat.cols, 3);

    Eigen::Matrix3f eigOut = ORB_SLAM3::Converter::toMatrix3f(cvMat);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(eigIn(r, c), eigOut(r, c), 1e-5f);
}

// ---------------------------------------------------------------------------
// toCvMat(Eigen::Vector3f) roundtrip with toVector3f
// ---------------------------------------------------------------------------
TEST(Converter, Vector3f_Roundtrip)
{
    Eigen::Vector3f vIn(1.1f, 2.2f, 3.3f);
    cv::Mat cvMat = ORB_SLAM3::Converter::toCvMat(vIn);

    ASSERT_EQ(cvMat.rows, 3);
    ASSERT_EQ(cvMat.cols, 1);

    Eigen::Vector3f vOut = ORB_SLAM3::Converter::toVector3f(cvMat);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(vIn(i), vOut(i), 1e-5f);
}

// ---------------------------------------------------------------------------
// toSE3Quat(cv::Mat): 4x4 identity -> g2o::SE3Quat should be identity
// ---------------------------------------------------------------------------
TEST(Converter, toSE3Quat_Identity)
{
    cv::Mat I = cv::Mat::eye(4, 4, CV_32F);
    g2o::SE3Quat se3 = ORB_SLAM3::Converter::toSE3Quat(I);

    // Translation should be zero
    Eigen::Vector3d t = se3.translation();
    EXPECT_NEAR(t(0), 0.0, 1e-6);
    EXPECT_NEAR(t(1), 0.0, 1e-6);
    EXPECT_NEAR(t(2), 0.0, 1e-6);

    // Rotation should be identity quaternion (w=1, x=y=z=0)
    Eigen::Quaterniond q = se3.rotation();
    EXPECT_NEAR(std::abs(q.w()), 1.0, 1e-6);
    EXPECT_NEAR(q.x(), 0.0, 1e-6);
    EXPECT_NEAR(q.y(), 0.0, 1e-6);
    EXPECT_NEAR(q.z(), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// tocvSkewMatrix: [1,2,3] -> verify anti-symmetric
// ---------------------------------------------------------------------------
TEST(Converter, tocvSkewMatrix_AntiSymmetric)
{
    cv::Mat v = (cv::Mat_<float>(3, 1) << 1.0f, 2.0f, 3.0f);
    cv::Mat skew = ORB_SLAM3::Converter::tocvSkewMatrix(v);

    ASSERT_EQ(skew.rows, 3);
    ASSERT_EQ(skew.cols, 3);

    // Anti-symmetric: S + S^T = 0
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(skew.at<float>(r, c) + skew.at<float>(c, r), 0.0f, 1e-6f);

    // Verify specific entries: skew([a,b,c]) = [[0,-c,b],[c,0,-a],[-b,a,0]]
    EXPECT_NEAR(skew.at<float>(0, 1), -3.0f, 1e-6f);
    EXPECT_NEAR(skew.at<float>(0, 2),  2.0f, 1e-6f);
    EXPECT_NEAR(skew.at<float>(1, 0),  3.0f, 1e-6f);
    EXPECT_NEAR(skew.at<float>(1, 2), -1.0f, 1e-6f);
    EXPECT_NEAR(skew.at<float>(2, 0), -2.0f, 1e-6f);
    EXPECT_NEAR(skew.at<float>(2, 1),  1.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// isRotationMatrix: identity=true, random=false
// ---------------------------------------------------------------------------
TEST(Converter, isRotationMatrix_Identity)
{
    cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
    EXPECT_TRUE(ORB_SLAM3::Converter::isRotationMatrix(I));
}

TEST(Converter, isRotationMatrix_RandomFalse)
{
    cv::Mat notRot = (cv::Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    EXPECT_FALSE(ORB_SLAM3::Converter::isRotationMatrix(notRot));
}

// ---------------------------------------------------------------------------
// toEuler(cv::Mat): identity rotation -> (0,0,0)
// ---------------------------------------------------------------------------
TEST(Converter, toEuler_Identity)
{
    cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
    std::vector<float> euler = ORB_SLAM3::Converter::toEuler(I);

    ASSERT_EQ(euler.size(), static_cast<std::size_t>(3));
    EXPECT_NEAR(euler[0], 0.0f, 1e-6f);
    EXPECT_NEAR(euler[1], 0.0f, 1e-6f);
    EXPECT_NEAR(euler[2], 0.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// NormalizeRotation(Eigen::Matrix3f): already normalized -> no change
// ---------------------------------------------------------------------------
TEST(Converter, NormalizeRotation_AlreadyNormalized)
{
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f result = ORB_SLAM3::IMU::NormalizeRotation(I);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(result(r, c), I(r, c), 1e-5f);
}
