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

#ifndef OPTIM_RIG_H
#define OPTIM_RIG_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3
{

    class GeometricCamera;

    namespace optim
    {

        // What an observation is projected with. Three kinds of observation
        // read three parts of it: one in the first camera, `camera`; one of a
        // rectified pair, the pinhole fx, fy, cx, cy and the baseline times fx;
        // one in the second camera of a pair that is not rectified, `camera2`
        // and where it is from the first.
        struct Rig
        {
            GeometricCamera* camera = nullptr;
            GeometricCamera* camera2 = nullptr;
            Eigen::Quaterniond Rrl = Eigen::Quaterniond::Identity();
            Eigen::Vector3d trl = Eigen::Vector3d::Zero();
            double fx = 0, fy = 0, cx = 0, cy = 0, bf = 0;

            bool operator==(const Rig &other) const
            {
                return camera == other.camera && camera2 == other.camera2 && Rrl.coeffs() == other.Rrl.coeffs() &&
                       trl == other.trl && fx == other.fx && fy == other.fy && cx == other.cx && cy == other.cy &&
                       bf == other.bf;
            }
        };

        // Which part of the rig an observation is in, and with it how many
        // numbers it is: (u, v), or (u, v, uR) for kStereo.
        enum ObservationKind : unsigned char
        {
            kMono,   // (u, v) in `camera`
            kStereo, // (u, v, uR) of a rectified pair: fx, fy, cx, cy, bf
            kRight,  // (u, v) in `camera2`, which is right-from-left of `camera`
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_RIG_H
