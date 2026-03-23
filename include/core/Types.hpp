#pragma once

#include "KeyFrame.hpp"
#include "g2o/types/types_seven_dof_expmap.h"
#include <map>
#include <set>
#include <Eigen/Core>

namespace ORB_SLAM3 {

// Shared type aliases used across modules (LoopClosing, Optimizer, etc.)
using KeyFrameAndPose = std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
    Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>>>;

using ConsistentGroup = std::pair<std::set<KeyFrame*>, int>;

} // namespace ORB_SLAM3
