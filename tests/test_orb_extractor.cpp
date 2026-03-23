/**
 * Tier 1 GTest: ORB feature extraction
 */

#include <gtest/gtest.h>
#include "ORBextractor.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Constructor: verify nlevels, scaleFactor, GetScaleFactors() size
// ---------------------------------------------------------------------------
TEST(ORBExtractor, ConstructorProperties)
{
    int nfeatures   = 1000;
    float scaleFactor = 1.2f;
    int nlevels     = 8;
    int iniThFAST   = 20;
    int minThFAST   = 7;

    ORB_SLAM3::ORBextractor extractor(nfeatures, scaleFactor, nlevels,
                                       iniThFAST, minThFAST);

    EXPECT_EQ(extractor.GetLevels(), nlevels);
    EXPECT_FLOAT_EQ(extractor.GetScaleFactor(), scaleFactor);

    std::vector<float> scaleFactors = extractor.GetScaleFactors();
    ASSERT_EQ(static_cast<int>(scaleFactors.size()), nlevels);

    // Level 0 scale factor should be 1.0
    EXPECT_FLOAT_EQ(scaleFactors[0], 1.0f);

    // Level 1 scale factor should be scaleFactor
    EXPECT_NEAR(scaleFactors[1], scaleFactor, 1e-5f);
}

// ---------------------------------------------------------------------------
// GetInverseScaleFactors: product with ScaleFactors approx 1.0
// ---------------------------------------------------------------------------
TEST(ORBExtractor, InverseScaleFactorsProduct)
{
    ORB_SLAM3::ORBextractor extractor(1000, 1.2f, 8, 20, 7);

    std::vector<float> scales    = extractor.GetScaleFactors();
    std::vector<float> invScales = extractor.GetInverseScaleFactors();

    ASSERT_EQ(scales.size(), invScales.size());

    for (std::size_t i = 0; i < scales.size(); ++i)
    {
        EXPECT_NEAR(scales[i] * invScales[i], 1.0f, 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// operator() on synthetic image with shapes: should extract > 0 keypoints
// ---------------------------------------------------------------------------
TEST(ORBExtractor, SyntheticImageExtractsKeypoints)
{
    ORB_SLAM3::ORBextractor extractor(500, 1.2f, 8, 20, 7);

    // Create a 640x480 grayscale image with drawn shapes for corners/features
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);

    // Draw rectangles, circles, and lines to create corner features
    cv::rectangle(image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255), 2);
    cv::rectangle(image, cv::Point(300, 100), cv::Point(500, 300), cv::Scalar(200), 3);
    cv::circle(image, cv::Point(400, 350), 80, cv::Scalar(180), 2);
    cv::line(image, cv::Point(100, 400), cv::Point(600, 350), cv::Scalar(220), 2);
    cv::rectangle(image, cv::Point(10, 300), cv::Point(150, 450), cv::Scalar(128), -1);
    // Add some small squares to create more corner features
    for (int i = 0; i < 10; ++i)
    {
        int x = 50 + i * 55;
        int y = 250;
        cv::rectangle(image, cv::Point(x, y), cv::Point(x + 20, y + 20), cv::Scalar(255), -1);
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> lappingArea = {0, 0};

    int nKps = extractor(image, cv::noArray(), keypoints, descriptors, lappingArea);

    EXPECT_GT(nKps, 0);
    EXPECT_EQ(nKps, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, 32);
}

// ---------------------------------------------------------------------------
// operator() on empty/black image: should extract 0 keypoints
// ---------------------------------------------------------------------------
TEST(ORBExtractor, BlackImageExtractsZero)
{
    ORB_SLAM3::ORBextractor extractor(500, 1.2f, 8, 20, 7);

    cv::Mat blackImage = cv::Mat::zeros(480, 640, CV_8UC1);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> lappingArea = {0, 0};

    int nKps = extractor(blackImage, cv::noArray(), keypoints, descriptors, lappingArea);

    EXPECT_EQ(nKps, 0);
    EXPECT_TRUE(keypoints.empty());
}

// ---------------------------------------------------------------------------
// ExtractorNode::DivideNode: split region, verify 4 children
// ---------------------------------------------------------------------------
TEST(ORBExtractor, DivideNode)
{
    ORB_SLAM3::ExtractorNode parent;
    parent.UL = cv::Point2i(0, 0);
    parent.UR = cv::Point2i(100, 0);
    parent.BL = cv::Point2i(0, 100);
    parent.BR = cv::Point2i(100, 100);

    // Populate with scattered keypoints
    for (int i = 0; i < 20; ++i)
    {
        cv::KeyPoint kp;
        kp.pt = cv::Point2f(static_cast<float>(i * 5),
                             static_cast<float>(i * 5));
        parent.vKeys.push_back(kp);
    }

    ORB_SLAM3::ExtractorNode n1, n2, n3, n4;
    parent.DivideNode(n1, n2, n3, n4);

    // Each child should cover a quadrant
    // n1: upper-left, n2: upper-right, n3: bottom-left, n4: bottom-right
    int halfX = (parent.UL.x + parent.UR.x) / 2;
    int halfY = (parent.UL.y + parent.BL.y) / 2;

    EXPECT_EQ(n1.UL, cv::Point2i(parent.UL.x, parent.UL.y));
    EXPECT_EQ(n1.BR, cv::Point2i(halfX, halfY));

    EXPECT_EQ(n2.UL, cv::Point2i(halfX, parent.UL.y));
    EXPECT_EQ(n2.BR, cv::Point2i(parent.UR.x, halfY));

    EXPECT_EQ(n3.UL, cv::Point2i(parent.UL.x, halfY));
    EXPECT_EQ(n3.BR, cv::Point2i(halfX, parent.BL.y));

    EXPECT_EQ(n4.UL, cv::Point2i(halfX, halfY));
    EXPECT_EQ(n4.BR, cv::Point2i(parent.BR.x, parent.BR.y));

    // All keypoints should be distributed among the children
    std::size_t totalKps = n1.vKeys.size() + n2.vKeys.size()
                         + n3.vKeys.size() + n4.vKeys.size();
    EXPECT_EQ(totalKps, parent.vKeys.size());
}
