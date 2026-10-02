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

#ifndef OPTIM_POSE4DOFGRAPHPROBLEM_H
#define OPTIM_POSE4DOFGRAPHPROBLEM_H

#include "optim/BodyPose.hpp"

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <cstddef>
#include <vector>

namespace ORB_SLAM3
{

    namespace optim
    {

        // The essential graph of an inertial map. Gravity fixes roll and pitch,
        // so a pose moves in translation and in yaw about the world's vertical
        // only. A constraint from pose i to pose j says what Tci_w * Tw_cj was,
        // in the first camera's frame, and all have the same information.
        //
        // The poses that are not fixed are laid out in the order of the arrays
        // and the constraints summed in the order of theirs.
        struct Pose4DofGraphProblem
        {
            // In: where to start. Out: what was found.
            std::vector<BodyPose, Eigen::aligned_allocator<BodyPose>> poses;
            std::vector<unsigned char> fixed;

            // One entry per constraint.
            std::vector<int> from;
            std::vector<int> to;
            std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> Rij;
            std::vector<Eigen::Vector3d> tij;
            Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();

            std::size_t constraints() const { return from.size(); }

            int addPose(const BodyPose &pose, bool bFixed)
            {
                poses.push_back(pose);
                fixed.push_back(bFixed);
                return static_cast<int>(poses.size()) - 1;
            }

            void addConstraint(int i, int j, const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation)
            {
                from.push_back(i);
                to.push_back(j);
                Rij.push_back(rotation);
                tij.push_back(translation);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_POSE4DOFGRAPHPROBLEM_H
