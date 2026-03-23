#pragma once

#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Tracking;

// Observer interface for receiving tracking state updates.
// Decouples Tracking from specific visualization implementations.
class TrackingObserver {
public:
    virtual ~TrackingObserver() = default;

    // Called after each frame is tracked
    virtual void onTrackingUpdate(Tracking* tracker) = 0;

    // Called when camera pose is updated
    virtual void onPoseUpdate(const Sophus::SE3f& pose) = 0;
};

} // namespace ORB_SLAM3
