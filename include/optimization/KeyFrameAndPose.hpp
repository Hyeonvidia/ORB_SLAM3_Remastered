/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef KEYFRAMEANDPOSE_H
#define KEYFRAMEANDPOSE_H

#include <orbslam3r/g2o_ext/compat.hpp>

#include <Eigen/Core>

#include <functional>
#include <map>
#include <utility>

namespace ORB_SLAM3
{

    class KeyFrame;

    // Keyframe -> Sim(3) pose, the currency of loop correction and map merging:
    // LoopClosing builds these and the Optimizer consumes them.
    //
    // It used to be LoopClosing::KeyFrameAndPose, nested in the class that
    // produces it, and that nesting was the whole of the optimization ->
    // loop_closing dependency -- 22 uses in Optimizer, and nothing else from
    // LoopClosing at all. At namespace scope in optimization/ the dependency
    // points the right way: loop_closing uses the optimizer, not the reverse.
    // LoopClosing keeps its nested name as an alias of this one.
    typedef std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
                     Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>>>
        KeyFrameAndPose;

} // namespace ORB_SLAM3

#endif // KEYFRAMEANDPOSE_H
