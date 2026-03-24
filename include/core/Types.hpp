#pragma once

#include "KeyFrame.hpp"
#include "core/Sim3Type.hpp"
#include <map>
#include <set>
#include <Eigen/Core>

namespace ORB_SLAM3 {

// Shared type aliases used across modules (LoopClosing, Optimizer, etc.)
using KeyFrameAndPose = std::map<KeyFrame*, Sim3Type, std::less<KeyFrame*>,
    Eigen::aligned_allocator<std::pair<KeyFrame* const, Sim3Type>>>;

using ConsistentGroup = std::pair<std::set<KeyFrame*>, int>;

} // namespace ORB_SLAM3
