/**
 * Tier 1 GTest: DBoW2 wrapper utilities
 */

#include <gtest/gtest.h>
#include "dbow2_wrapper.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

// ---------------------------------------------------------------------------
// toDescriptorVector: cv::Mat -> vector conversion
// ---------------------------------------------------------------------------
TEST(DBoW2Wrapper, toDescriptorVector)
{
    cv::Mat descriptors(7, 32, CV_8UC1);
    cv::randu(descriptors, cv::Scalar(0), cv::Scalar(255));

    std::vector<cv::Mat> vDesc = slam3::dbow2::toDescriptorVector(descriptors);

    ASSERT_EQ(vDesc.size(), static_cast<std::size_t>(7));
    for (int i = 0; i < 7; ++i)
    {
        EXPECT_EQ(vDesc[i].rows, 1);
        EXPECT_EQ(vDesc[i].cols, 32);
        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(vDesc[i].at<unsigned char>(0, j),
                       descriptors.at<unsigned char>(i, j));
        }
    }
}

// ---------------------------------------------------------------------------
// toDescriptorVector: empty Mat -> empty vector
// ---------------------------------------------------------------------------
TEST(DBoW2Wrapper, toDescriptorVector_Empty)
{
    cv::Mat empty;
    std::vector<cv::Mat> vDesc = slam3::dbow2::toDescriptorVector(empty);
    EXPECT_TRUE(vDesc.empty());
}

// ---------------------------------------------------------------------------
// descriptorDistance: identical descriptors -> 0
// ---------------------------------------------------------------------------
TEST(DBoW2Wrapper, DescriptorDistance_Identical)
{
    cv::Mat desc(1, 32, CV_8UC1);
    cv::randu(desc, cv::Scalar(0), cv::Scalar(255));

    int dist = slam3::dbow2::descriptorDistance(desc, desc);
    EXPECT_EQ(dist, 0);
}

// ---------------------------------------------------------------------------
// descriptorDistance: all zeros vs all 0xFF -> 256
// ---------------------------------------------------------------------------
TEST(DBoW2Wrapper, DescriptorDistance_AllDifferent)
{
    cv::Mat zeros = cv::Mat::zeros(1, 32, CV_8UC1);
    cv::Mat ones(1, 32, CV_8UC1, cv::Scalar(0xFF));

    int dist = slam3::dbow2::descriptorDistance(zeros, ones);
    EXPECT_EQ(dist, 256);
}

// ---------------------------------------------------------------------------
// descriptorDistance: known single-bit difference
// ---------------------------------------------------------------------------
TEST(DBoW2Wrapper, DescriptorDistance_SingleBit)
{
    cv::Mat a = cv::Mat::zeros(1, 32, CV_8UC1);
    cv::Mat b = cv::Mat::zeros(1, 32, CV_8UC1);
    b.at<unsigned char>(0, 15) = 0x80; // 1 bit set in byte 15

    int dist = slam3::dbow2::descriptorDistance(a, b);
    EXPECT_EQ(dist, 1);
}

// ---------------------------------------------------------------------------
// descriptorDistance: symmetry
// ---------------------------------------------------------------------------
TEST(DBoW2Wrapper, DescriptorDistance_Symmetry)
{
    cv::Mat a(1, 32, CV_8UC1);
    cv::Mat b(1, 32, CV_8UC1);
    cv::randu(a, cv::Scalar(0), cv::Scalar(255));
    cv::randu(b, cv::Scalar(0), cv::Scalar(255));

    int distAB = slam3::dbow2::descriptorDistance(a, b);
    int distBA = slam3::dbow2::descriptorDistance(b, a);

    EXPECT_EQ(distAB, distBA);
}
