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

#ifndef OPTIM_INERTIALALIGNMENTPROBLEM_H
#define OPTIM_INERTIALALIGNMENTPROBLEM_H

#include "optim/BodyPose.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace ORB_SLAM3
{

    namespace IMU
    {
        class Preintegrated;
    }

    namespace optim
    {

        // What ties a map built by vision to what the inertial unit measured:
        // the three Optimizer::InertialOptimization. The keyframes' poses are
        // held as vision found them, up to the scale of the map; what is found
        // is that scale, which way gravity points, the velocity at each
        // keyframe and one pair of biases for all of them -- or those of them
        // that are not held too.
        struct InertialAlignmentProblem
        {
            // Held. In the order their velocities are to be laid out in.
            std::vector<BodyPose> poses;

            // In: where to start. Out: what was found.
            std::vector<Eigen::Vector3d> velocity; // one per pose
            Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
            Eigen::Vector3d accBias = Eigen::Vector3d::Zero();
            Eigen::Matrix3d Rwg = Eigen::Matrix3d::Identity(); // gravity is -z of g
            double scale = 1.0;

            bool velocitiesFixed = false;
            bool biasesFixed = false;
            bool gravityFixed = false;
            bool scaleFixed = false;

            // Each bias held towards zero, its information this times
            // identity. Summed before the inertial terms, accelerometer first.
            bool biasPriors = false;
            double accPriorInformation = 0.0;
            double gyroPriorInformation = 0.0;

            // One entry per inertial term, in the order they are to be summed:
            // what was measured between two poses, integrated. The
            // preintegrations are not the problem's: whoever builds it keeps
            // them for as long as it is solved.
            std::vector<int> from;
            std::vector<int> to;
            std::vector<IMU::Preintegrated*> preintegration;
            double huber = 0.0; // the robust cost's width; 0 for none

            std::size_t constraints() const { return from.size(); }

            int addPose(const BodyPose &pose, const Eigen::Vector3d &v)
            {
                poses.push_back(pose);
                velocity.push_back(v);
                return static_cast<int>(poses.size()) - 1;
            }

            void addConstraint(int i, int j, IMU::Preintegrated* pInt)
            {
                from.push_back(i);
                to.push_back(j);
                preintegration.push_back(pInt);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_INERTIALALIGNMENTPROBLEM_H
