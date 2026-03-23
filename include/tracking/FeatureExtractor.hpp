#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM3 {

// =============================================================================
// FeatureExtractor — Abstract interface for feature extraction
// Allows swapping ORB with other methods (SuperPoint, SIFT, etc.)
// =============================================================================
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    // Extract keypoints and descriptors from image
    virtual int extract(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::OutputArray descriptors,
                        std::vector<int>& lappingArea) = 0;

    // Scale pyramid info
    virtual int GetLevels() const = 0;
    virtual float GetScaleFactor() const = 0;
    virtual std::vector<float> GetScaleFactors() const = 0;
    virtual std::vector<float> GetInverseScaleFactors() const = 0;
    virtual std::vector<float> GetScaleSigmaSquares() const = 0;
    virtual std::vector<float> GetInverseScaleSigmaSquares() const = 0;

    // Image pyramid access
    virtual const std::vector<cv::Mat>& getImagePyramid() const = 0;
};

// =============================================================================
// KeypointDistributor — Strategy for spatial keypoint distribution
// =============================================================================
class KeypointDistributor {
public:
    virtual ~KeypointDistributor() = default;

    virtual std::vector<cv::KeyPoint> distribute(
        const std::vector<cv::KeyPoint>& candidates,
        int minX, int maxX, int minY, int maxY,
        int nFeatures, int level) = 0;
};

// =============================================================================
// OctTreeDistributor — Quadtree-based even distribution (original ORB-SLAM3)
// =============================================================================
class OctTreeDistributor : public KeypointDistributor {
public:
    std::vector<cv::KeyPoint> distribute(
        const std::vector<cv::KeyPoint>& candidates,
        int minX, int maxX, int minY, int maxY,
        int nFeatures, int level) override;
};

// =============================================================================
// GridDistributor — Simple grid-based distribution (alternative)
// =============================================================================
class GridDistributor : public KeypointDistributor {
public:
    std::vector<cv::KeyPoint> distribute(
        const std::vector<cv::KeyPoint>& candidates,
        int minX, int maxX, int minY, int maxY,
        int nFeatures, int level) override;
};

} // namespace ORB_SLAM3
