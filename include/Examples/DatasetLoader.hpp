#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "ImuTypes.hpp"

namespace ORB_SLAM3 {

// =============================================================================
// DatasetLoader — Abstract interface for dataset loading
// =============================================================================
class DatasetLoader {
public:
    virtual ~DatasetLoader() = default;

    // Load image paths and timestamps for a sequence
    virtual void loadSequence(int seqIdx = 0) = 0;

    // Number of images in current sequence
    virtual int numImages() const = 0;

    // Number of sequences
    virtual int numSequences() const = 0;

    // Get left image at index
    virtual cv::Mat getImage(int idx) const = 0;

    // Get right image at index (stereo only, default empty)
    virtual cv::Mat getRightImage(int idx) const { return cv::Mat(); }

    // Get depth image at index (RGBD only, default empty)
    virtual cv::Mat getDepthImage(int idx) const { return cv::Mat(); }

    // Get timestamp at index
    virtual double getTimestamp(int idx) const = 0;

    // Get IMU measurements between frames (inertial only)
    virtual std::vector<IMU::Point> getImuBetween(int prevIdx, int currIdx) const { return {}; }

    // Whether this dataset has IMU data
    virtual bool hasImu() const { return false; }

    // Trajectory save format name
    virtual std::string trajectoryFormat() const = 0;
};

// =============================================================================
// EurocLoader — EuRoC MAV dataset
// =============================================================================
class EurocLoader : public DatasetLoader {
public:
    EurocLoader(const std::vector<std::string>& seqPaths,
                const std::vector<std::string>& timestampPaths,
                bool loadImu = false)
        : seqPaths_(seqPaths), timestampPaths_(timestampPaths), loadImu_(loadImu) {}

    void loadSequence(int seqIdx = 0) override {
        curSeq_ = seqIdx;
        imageFiles_.clear();
        timestamps_.clear();

        std::ifstream f(timestampPaths_[seqIdx]);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string timestamp;
            ss >> timestamp;
            imageFiles_.push_back(seqPaths_[seqIdx] + "/mav0/cam0/data/" + timestamp + ".png");
            timestamps_.push_back(std::stod(timestamp) * 1e-9);
        }

        if (loadImu_) loadImuData(seqIdx);
    }

    int numImages() const override { return static_cast<int>(imageFiles_.size()); }
    int numSequences() const override { return static_cast<int>(seqPaths_.size()); }

    cv::Mat getImage(int idx) const override {
        return cv::imread(imageFiles_[idx], cv::IMREAD_UNCHANGED);
    }

    cv::Mat getRightImage(int idx) const override {
        std::string rightPath = imageFiles_[idx];
        size_t pos = rightPath.find("cam0");
        if (pos != std::string::npos)
            rightPath.replace(pos, 4, "cam1");
        return cv::imread(rightPath, cv::IMREAD_UNCHANGED);
    }

    double getTimestamp(int idx) const override { return timestamps_[idx]; }

    bool hasImu() const override { return loadImu_; }

    std::vector<IMU::Point> getImuBetween(int prevIdx, int currIdx) const override {
        if (!loadImu_ || imuData_.empty()) return {};
        double t0 = (prevIdx >= 0) ? timestamps_[prevIdx] : 0.0;
        double t1 = timestamps_[currIdx];
        std::vector<IMU::Point> result;
        for (const auto& p : imuData_) {
            if (p.t >= t0 && p.t <= t1)
                result.push_back(p);
        }
        return result;
    }

    std::string trajectoryFormat() const override { return "euroc"; }

private:
    void loadImuData(int seqIdx) {
        imuData_.clear();
        std::string imuPath = seqPaths_[seqIdx] + "/mav0/imu0/data.csv";
        std::ifstream f(imuPath);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss(line);
            double t, wx, wy, wz, ax, ay, az;
            ss >> t >> wx >> wy >> wz >> ax >> ay >> az;
            imuData_.emplace_back(static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(az),
                                  static_cast<float>(wx), static_cast<float>(wy), static_cast<float>(wz),
                                  t * 1e-9);
        }
    }

    std::vector<std::string> seqPaths_;
    std::vector<std::string> timestampPaths_;
    bool loadImu_;
    int curSeq_ = 0;
    std::vector<std::string> imageFiles_;
    std::vector<double> timestamps_;
    std::vector<IMU::Point> imuData_;
};

// =============================================================================
// KittiLoader — KITTI odometry dataset
// =============================================================================
class KittiLoader : public DatasetLoader {
public:
    explicit KittiLoader(const std::string& sequencePath)
        : seqPath_(sequencePath) {}

    void loadSequence(int seqIdx = 0) override {
        imageFiles_.clear();
        timestamps_.clear();

        // Load timestamps
        std::ifstream fTimes(seqPath_ + "/times.txt");
        std::string line;
        while (std::getline(fTimes, line)) {
            if (line.empty()) continue;
            timestamps_.push_back(std::stod(line));
        }

        // Generate image filenames
        for (int i = 0; i < static_cast<int>(timestamps_.size()); i++) {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(6) << i;
            imageFiles_.push_back(seqPath_ + "/image_0/" + ss.str() + ".png");
            rightFiles_.push_back(seqPath_ + "/image_1/" + ss.str() + ".png");
        }
    }

    int numImages() const override { return static_cast<int>(imageFiles_.size()); }
    int numSequences() const override { return 1; }

    cv::Mat getImage(int idx) const override {
        return cv::imread(imageFiles_[idx], cv::IMREAD_UNCHANGED);
    }

    cv::Mat getRightImage(int idx) const override {
        return cv::imread(rightFiles_[idx], cv::IMREAD_UNCHANGED);
    }

    double getTimestamp(int idx) const override { return timestamps_[idx]; }

    std::string trajectoryFormat() const override { return "kitti"; }

private:
    std::string seqPath_;
    std::vector<std::string> imageFiles_;
    std::vector<std::string> rightFiles_;
    std::vector<double> timestamps_;
};

// =============================================================================
// TumLoader — TUM RGB-D dataset
// =============================================================================
class TumLoader : public DatasetLoader {
public:
    TumLoader(const std::string& basePath, const std::string& associationFile)
        : basePath_(basePath) {
        loadAssociations(associationFile);
    }

    void loadSequence(int seqIdx = 0) override { /* already loaded in constructor */ }

    int numImages() const override { return static_cast<int>(rgbFiles_.size()); }
    int numSequences() const override { return 1; }

    cv::Mat getImage(int idx) const override {
        return cv::imread(basePath_ + "/" + rgbFiles_[idx], cv::IMREAD_UNCHANGED);
    }

    cv::Mat getDepthImage(int idx) const override {
        return cv::imread(basePath_ + "/" + depthFiles_[idx], cv::IMREAD_UNCHANGED);
    }

    double getTimestamp(int idx) const override { return timestamps_[idx]; }

    std::string trajectoryFormat() const override { return "tum"; }

private:
    void loadAssociations(const std::string& file) {
        std::ifstream f(file);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            double tRgb, tDepth;
            std::string rgbFile, depthFile;
            ss >> tRgb >> rgbFile >> tDepth >> depthFile;
            timestamps_.push_back(tRgb);
            rgbFiles_.push_back(rgbFile);
            depthFiles_.push_back(depthFile);
        }
    }

    std::string basePath_;
    std::vector<std::string> rgbFiles_;
    std::vector<std::string> depthFiles_;
    std::vector<double> timestamps_;
};

} // namespace ORB_SLAM3
