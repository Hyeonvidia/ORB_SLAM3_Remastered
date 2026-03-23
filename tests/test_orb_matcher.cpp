/**
 * Tier 1 GTest: ORB descriptor distance
 */

#include <gtest/gtest.h>
#include "ORBmatcher.hpp"

#include <opencv2/core/core.hpp>

// ---------------------------------------------------------------------------
// DescriptorDistance: same descriptor -> 0
// ---------------------------------------------------------------------------
TEST(ORBMatcher, DescriptorDistance_Same)
{
    cv::Mat desc(1, 32, CV_8UC1);
    cv::randu(desc, cv::Scalar(0), cv::Scalar(255));

    int dist = ORB_SLAM3::ORBmatcher::DescriptorDistance(desc, desc);
    EXPECT_EQ(dist, 0);
}

// ---------------------------------------------------------------------------
// DescriptorDistance: all zeros vs all ones -> 256
// ---------------------------------------------------------------------------
TEST(ORBMatcher, DescriptorDistance_AllZerosVsAllOnes)
{
    cv::Mat zeros = cv::Mat::zeros(1, 32, CV_8UC1);
    cv::Mat ones(1, 32, CV_8UC1, cv::Scalar(0xFF));

    int dist = ORB_SLAM3::ORBmatcher::DescriptorDistance(zeros, ones);
    EXPECT_EQ(dist, 256);
}

// ---------------------------------------------------------------------------
// DescriptorDistance: known bit pattern
// Descriptor a: all zeros
// Descriptor b: first byte 0x01 (1 bit set), rest zeros -> distance = 1
// ---------------------------------------------------------------------------
TEST(ORBMatcher, DescriptorDistance_KnownPattern)
{
    cv::Mat a = cv::Mat::zeros(1, 32, CV_8UC1);
    cv::Mat b = cv::Mat::zeros(1, 32, CV_8UC1);

    b.at<unsigned char>(0, 0) = 0x01; // 1 bit different

    int dist = ORB_SLAM3::ORBmatcher::DescriptorDistance(a, b);
    EXPECT_EQ(dist, 1);

    // 0xFF = 8 bits set in one byte
    cv::Mat c = cv::Mat::zeros(1, 32, CV_8UC1);
    c.at<unsigned char>(0, 0) = 0xFF;

    dist = ORB_SLAM3::ORBmatcher::DescriptorDistance(a, c);
    EXPECT_EQ(dist, 8);
}

// ---------------------------------------------------------------------------
// DescriptorDistance: symmetry (a,b) == (b,a)
// ---------------------------------------------------------------------------
TEST(ORBMatcher, DescriptorDistance_Symmetry)
{
    cv::Mat a(1, 32, CV_8UC1);
    cv::Mat b(1, 32, CV_8UC1);
    cv::randu(a, cv::Scalar(0), cv::Scalar(255));
    cv::randu(b, cv::Scalar(0), cv::Scalar(255));

    int distAB = ORB_SLAM3::ORBmatcher::DescriptorDistance(a, b);
    int distBA = ORB_SLAM3::ORBmatcher::DescriptorDistance(b, a);

    EXPECT_EQ(distAB, distBA);
}
