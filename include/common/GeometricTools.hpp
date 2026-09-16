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

#pragma once

#include <Eigen/Core>

namespace ORB_SLAM3::GeometricTools
{

/// Triangulates a point seen by two cameras, by the linear (DLT) method.
///
/// @param x_c1   bearing of the observation in camera 1, normalised so z == 1
/// @param x_c2   the same for camera 2
/// @param Tc1w   3x4 world-to-camera-1 transform
/// @param Tc2w   3x4 world-to-camera-2 transform
/// @param x3D    the triangulated point, in world coordinates
/// @return false when the solution is at infinity, leaving x3D untouched
bool Triangulate(const Eigen::Vector3f &x_c1, const Eigen::Vector3f &x_c2, const Eigen::Matrix<float, 3, 4> &Tc1w,
                 const Eigen::Matrix<float, 3, 4> &Tc2w, Eigen::Vector3f &x3D);

} // namespace ORB_SLAM3::GeometricTools
