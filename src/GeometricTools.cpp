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

#include "GeometricTools.hpp"

#include <Eigen/SVD>

namespace ORB_SLAM3::GeometricTools {

bool Triangulate(const Eigen::Vector3f &x_c1, const Eigen::Vector3f &x_c2,
                 const Eigen::Matrix<float, 3, 4> &Tc1w,
                 const Eigen::Matrix<float, 3, 4> &Tc2w,
                 Eigen::Vector3f &x3D) {
  // Each observation contributes two rows of the form x * P_row2 - P_row0,
  // so the triangulated point is the null space of A.
  Eigen::Matrix4f A;
  A.block<1, 4>(0, 0) = x_c1(0) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(0, 0);
  A.block<1, 4>(1, 0) = x_c1(1) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(1, 0);
  A.block<1, 4>(2, 0) = x_c2(0) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(0, 0);
  A.block<1, 4>(3, 0) = x_c2(1) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(1, 0);

  const Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4f x3Dh = svd.matrixV().col(3);

  // A homogeneous w of zero puts the point at infinity: the two rays are
  // parallel and there is nothing to triangulate.
  if (x3Dh(3) == 0) {
    return false;
  }

  x3D = x3Dh.head(3) / x3Dh(3);
  return true;
}

}  // namespace ORB_SLAM3::GeometricTools
