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

#ifndef BODYPOSEOF_H
#define BODYPOSEOF_H

#include "optim/BodyPose.hpp"

#include <Eigen/Core>

namespace ORB_SLAM3
{

    class Frame;
    class KeyFrame;

    // The body's pose and its cameras, copied out of a keyframe or a frame for
    // an inertial optimisation: what v1.0's ImuCamPose read when it was made
    // from one.
    optim::BodyPose BodyPoseOf(KeyFrame* pKF);
    optim::BodyPose BodyPoseOf(Frame* pF);

    // For a pose graph: the keyframe's calibration with a pose given as the
    // first camera in the world. One camera only.
    optim::BodyPose BodyPoseOf(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, KeyFrame* pKF);

} // namespace ORB_SLAM3

#endif // BODYPOSEOF_H
