#pragma once

#include <vector>
#include "ImuTypes.hpp"

namespace ORB_SLAM3 {

class KeyFrame;

// =============================================================================
// Module interfaces — break circular dependencies between modules
// =============================================================================

// Interface for modules that consume keyframes (e.g., LoopClosing)
class IKeyFrameConsumer {
public:
    virtual ~IKeyFrameConsumer() = default;
    virtual void InsertKeyFrame(KeyFrame* pKF) = 0;
};

// Interface for controlling the mapping thread (e.g., from LoopClosing)
class IMappingControl {
public:
    virtual ~IMappingControl() = default;
    virtual void RequestStop() = 0;
    virtual bool Stop() = 0;
    virtual bool isStopped() = 0;
    virtual void Release() = 0;
    virtual bool SetNotStop(bool flag) = 0;
    virtual bool stopRequested() = 0;
    virtual bool AcceptKeyFrames() = 0;
    virtual void SetAcceptKeyFrames(bool flag) = 0;
    virtual void EmptyQueue() = 0;
    virtual bool IsInitializing() = 0;
    virtual double GetCurrKFTime() = 0;
    virtual KeyFrame* GetCurrKF() = 0;
};

// Interface for querying/updating tracking state (used by LocalMapping)
class ITrackingState {
public:
    virtual ~ITrackingState() = default;

    // Query
    virtual int GetMatchesInliers() = 0;
    virtual int GetTrackingState() const = 0;
    virtual double GetLastFrameTimestamp() const = 0;
    virtual double GetCurrentFrameTimestamp() const = 0;

    // Mutate
    virtual void SetTrackingState(int state) = 0;
    virtual void UpdateFrameIMU(float s, const IMU::Bias& b, KeyFrame* pCurrentKeyFrame) = 0;
    virtual void SetIMUStartTime(double t) = 0;
};

} // namespace ORB_SLAM3
