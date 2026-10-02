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

#ifndef CONSTRAINTPOSEIMU_H
#define CONSTRAINTPOSEIMU_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace ORB_SLAM3
{

    // What one inertial pose optimisation leaves known of a frame's state for
    // the next to start from: where the state was found and how firmly, H in
    // the order rotation, position, velocity, gyroscope bias, accelerometer
    // bias. A frame holds one (Frame::mpcpi). H is made symmetric and
    // positive semi-definite here.
    class ConstraintPoseImu
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConstraintPoseImu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                          const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_,
                          const Eigen::Matrix<double, 15, 15> &H_)
            : Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_)
        {
            H = (H + H) / 2;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> es(H);
            Eigen::Matrix<double, 15, 1> eigs = es.eigenvalues();
            for(int i = 0; i < 15; i++)
                if(eigs[i] < 1e-12)
                    eigs[i] = 0;
            H = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
        }

        Eigen::Matrix3d Rwb;
        Eigen::Vector3d twb;
        Eigen::Vector3d vwb;
        Eigen::Vector3d bg;
        Eigen::Vector3d ba;
        Eigen::Matrix<double, 15, 15> H;
    };

} // namespace ORB_SLAM3

#endif // CONSTRAINTPOSEIMU_H
