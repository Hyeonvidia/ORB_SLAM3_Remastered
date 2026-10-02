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

#ifndef OPTIM_SIM3GRAPHPROBLEM_H
#define OPTIM_SIM3GRAPHPROBLEM_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <vector>

namespace ORB_SLAM3
{

    namespace optim
    {

        // A graph of poses held together by what each was relative to another:
        // the essential graph Loop Closing optimises after a loop or a merge.
        // A pose is a similarity, keyframe-from-world; a constraint from pose
        // i to pose j says what Sj * Si^-1 was, and all weigh the same.
        //
        // The poses that are not fixed are laid out in the order of the
        // arrays and the constraints summed in the order of theirs.
        struct Sim3GraphProblem
        {
            // In: where to start. Out: what was found.
            std::vector<Eigen::Quaterniond> R;
            std::vector<Eigen::Vector3d> t;
            std::vector<double> s;
            std::vector<unsigned char> fixed;
            std::vector<unsigned char> fixScale; // the scale of this pose is held

            // One entry per constraint.
            std::vector<int> from;
            std::vector<int> to;
            std::vector<Eigen::Quaterniond> Rji;
            std::vector<Eigen::Vector3d> tji;
            std::vector<double> sji;

            std::size_t poses() const { return R.size(); }
            std::size_t constraints() const { return from.size(); }

            int addPose(const Eigen::Quaterniond &rotation, const Eigen::Vector3d &translation, double scale,
                        bool bFixed, bool bFixScale)
            {
                R.push_back(rotation);
                t.push_back(translation);
                s.push_back(scale);
                fixed.push_back(bFixed);
                fixScale.push_back(bFixScale);
                return static_cast<int>(R.size()) - 1;
            }

            void addConstraint(int i, int j, const Eigen::Quaterniond &rotation, const Eigen::Vector3d &translation,
                               double scale)
            {
                from.push_back(i);
                to.push_back(j);
                Rji.push_back(rotation);
                tji.push_back(translation);
                sji.push_back(scale);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_SIM3GRAPHPROBLEM_H
