#pragma once

#include "System.hpp"
#include "Examples/DatasetLoader.hpp"

#include <chrono>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unistd.h>

namespace ORB_SLAM3 {

// =============================================================================
// SlamRunner — Template Method pattern for SLAM execution
// =============================================================================
class SlamRunner {
public:
    SlamRunner(const std::string& vocPath, const std::string& settingsPath,
               DatasetLoader& loader, bool viewer = true)
        : vocPath_(vocPath), settingsPath_(settingsPath),
          loader_(loader), viewer_(viewer) {
        // Auto-disable viewer if no display available
        if (viewer_ && !std::getenv("DISPLAY"))
            viewer_ = false;
    }

    virtual ~SlamRunner() = default;

    // Template method — defines the skeleton algorithm
    int run() {
        // Create SLAM system
        System slam(vocPath_, settingsPath_, sensorType(), viewer_);
        float imageScale = slam.GetImageScale();

        for (int seq = 0; seq < loader_.numSequences(); seq++) {
            if (seq > 0) {
                slam.ChangeDataset();
            }
            loader_.loadSequence(seq);
            int nImages = loader_.numImages();
            std::vector<float> vTimesTrack(nImages, 0.0f);

            std::cout << "\n-------\nStart processing sequence " << seq << " ..." << std::endl;
            std::cout << "Images in the sequence: " << nImages << std::endl;

            for (int ni = 0; ni < nImages; ni++) {
                // Load frame data
                cv::Mat im = loader_.getImage(ni);
                double tframe = loader_.getTimestamp(ni);

                if (im.empty()) {
                    std::cerr << "Failed to load image: " << ni << std::endl;
                    continue;
                }

                // Scale if needed
                if (imageScale != 1.f) {
                    int width = im.cols * imageScale;
                    int height = im.rows * imageScale;
                    cv::resize(im, im, cv::Size(width, height));
                }

                // Time the tracking
                auto t1 = std::chrono::steady_clock::now();

                track(slam, im, tframe, ni);

                auto t2 = std::chrono::steady_clock::now();
                double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
                vTimesTrack[ni] = ttrack;

                // Frame synchronization
                double T = 0;
                if (ni < nImages - 1)
                    T = loader_.getTimestamp(ni + 1) - tframe;
                else if (ni > 0)
                    T = tframe - loader_.getTimestamp(ni - 1);

                if (ttrack < T)
                    usleep(static_cast<useconds_t>((T - ttrack) * 1e6));
            }

            printStats(vTimesTrack);
        }

        // Keep viewer open for a moment after processing
        if (viewer_) {
            std::cout << "\nProcessing complete. Viewer will stay open for 10 seconds..." << std::endl;
            usleep(10 * 1000000);
        }

        slam.Shutdown();
        saveTrajectory(slam);
        return 0;
    }

protected:
    // Pure virtual — sensor type for System constructor
    virtual System::eSensor sensorType() const = 0;

    // Pure virtual — dispatch to appropriate Track function
    virtual void track(System& slam, const cv::Mat& im, double timestamp, int frameIdx) = 0;

    // Virtual — save trajectory (override for different formats)
    virtual void saveTrajectory(System& slam) {
        const std::string fmt = loader_.trajectoryFormat();
        if (fmt == "euroc") {
            slam.SaveTrajectoryEuRoC("CameraTrajectory.txt");
            slam.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
        } else if (fmt == "tum") {
            slam.SaveTrajectoryTUM("CameraTrajectory.txt");
            slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        } else {
            slam.SaveTrajectoryKITTI("CameraTrajectory.txt");
            slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        }
    }

    void printStats(const std::vector<float>& vTimes) {
        if (vTimes.empty()) return;
        std::vector<float> sorted = vTimes;
        std::sort(sorted.begin(), sorted.end());
        float total = std::accumulate(sorted.begin(), sorted.end(), 0.0f);
        std::cout << "-------\n";
        std::cout << "median tracking time: " << sorted[sorted.size() / 2] << std::endl;
        std::cout << "mean tracking time: " << total / sorted.size() << std::endl;
    }

    DatasetLoader& loader_;
    std::string vocPath_;
    std::string settingsPath_;
    bool viewer_;
};

// =============================================================================
// MonoRunner — Monocular SLAM
// =============================================================================
class MonoRunner : public SlamRunner {
public:
    using SlamRunner::SlamRunner;

protected:
    System::eSensor sensorType() const override { return System::MONOCULAR; }

    void track(System& slam, const cv::Mat& im, double timestamp, int frameIdx) override {
        slam.TrackMonocular(im, timestamp);
    }
};

// =============================================================================
// MonoInertialRunner — Monocular-Inertial SLAM
// =============================================================================
class MonoInertialRunner : public SlamRunner {
public:
    using SlamRunner::SlamRunner;

protected:
    System::eSensor sensorType() const override { return System::IMU_MONOCULAR; }

    void track(System& slam, const cv::Mat& im, double timestamp, int frameIdx) override {
        auto imuMeas = loader_.getImuBetween(frameIdx - 1, frameIdx);
        slam.TrackMonocular(im, timestamp, imuMeas);
    }
};

// =============================================================================
// StereoRunner — Stereo SLAM
// =============================================================================
class StereoRunner : public SlamRunner {
public:
    using SlamRunner::SlamRunner;

protected:
    System::eSensor sensorType() const override { return System::STEREO; }

    void track(System& slam, const cv::Mat& im, double timestamp, int frameIdx) override {
        cv::Mat imRight = loader_.getRightImage(frameIdx);
        float imageScale = slam.GetImageScale();
        if (imageScale != 1.f) {
            int width = imRight.cols * imageScale;
            int height = imRight.rows * imageScale;
            cv::resize(imRight, imRight, cv::Size(width, height));
        }
        slam.TrackStereo(im, imRight, timestamp);
    }
};

// =============================================================================
// StereoInertialRunner — Stereo-Inertial SLAM
// =============================================================================
class StereoInertialRunner : public SlamRunner {
public:
    using SlamRunner::SlamRunner;

protected:
    System::eSensor sensorType() const override { return System::IMU_STEREO; }

    void track(System& slam, const cv::Mat& im, double timestamp, int frameIdx) override {
        cv::Mat imRight = loader_.getRightImage(frameIdx);
        float imageScale = slam.GetImageScale();
        if (imageScale != 1.f) {
            int width = imRight.cols * imageScale;
            int height = imRight.rows * imageScale;
            cv::resize(imRight, imRight, cv::Size(width, height));
        }
        auto imuMeas = loader_.getImuBetween(frameIdx - 1, frameIdx);
        slam.TrackStereo(im, imRight, timestamp, imuMeas);
    }
};

// =============================================================================
// RgbdRunner — RGB-D SLAM
// =============================================================================
class RgbdRunner : public SlamRunner {
public:
    using SlamRunner::SlamRunner;

protected:
    System::eSensor sensorType() const override { return System::RGBD; }

    void track(System& slam, const cv::Mat& im, double timestamp, int frameIdx) override {
        cv::Mat imDepth = loader_.getDepthImage(frameIdx);
        slam.TrackRGBD(im, imDepth, timestamp);
    }
};

} // namespace ORB_SLAM3
