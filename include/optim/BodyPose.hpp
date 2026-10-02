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

#ifndef OPTIM_BODYPOSE_H
#define OPTIM_BODYPOSE_H

#include <Eigen/Core>

namespace ORB_SLAM3
{

    class GeometricCamera;

    namespace optim
    {

        // The pose of a body that carries an inertial unit and one or two
        // cameras, as the inertial optimisations hold it: the body in the
        // world, and each camera from the world and from the body. What moves
        // in an optimisation is the body; the cameras follow.
        struct BodyPose
        {
            Eigen::Matrix3d Rwb = Eigen::Matrix3d::Identity();
            Eigen::Vector3d twb = Eigen::Vector3d::Zero();

            int nCameras = 1;
            Eigen::Matrix3d Rcw[2];
            Eigen::Vector3d tcw[2];
            Eigen::Matrix3d Rcb[2], Rbc[2];
            Eigen::Vector3d tcb[2], tbc[2];
            double bf = 0.0;
            GeometricCamera* camera[2] = {nullptr, nullptr};
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_BODYPOSE_H
