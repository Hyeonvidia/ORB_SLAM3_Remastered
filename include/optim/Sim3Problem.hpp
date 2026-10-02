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

#ifndef OPTIM_SIM3PROBLEM_H
#define OPTIM_SIM3PROBLEM_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <vector>

namespace ORB_SLAM3
{

    class GeometricCamera;

    namespace optim
    {

        // The similarity between two cameras from points each has and sees:
        // what Optimizer::OptimizeSim3 solves for a loop or a merge candidate.
        // S12 takes a point of camera 2 into camera 1. A pair is a point in
        // each camera's own frame, neither of which moves, and where each
        // camera saw the other's: the point of camera 2, taken by S12, in
        // camera 1, and the point of camera 1, taken back, in camera 2.
        struct Sim3Problem
        {
            // In: where to start. Out: what was found.
            Eigen::Quaterniond R12 = Eigen::Quaterniond::Identity();
            Eigen::Vector3d t12 = Eigen::Vector3d::Zero();
            double s12 = 1.0;
            bool fixScale = false; // the scale is held: stereo, RGB-D

            GeometricCamera* camera1 = nullptr;
            GeometricCamera* camera2 = nullptr;

            // One entry per pair, in the order they are to be summed.
            std::vector<Eigen::Vector3d> X1; // in camera 1's frame
            std::vector<Eigen::Vector3d> X2; // in camera 2's frame
            std::vector<Eigen::Vector2d> uv1;
            std::vector<Eigen::Vector2d> uv2;
            std::vector<double> invSigma2_1; // the information of each is this times identity
            std::vector<double> invSigma2_2;
            double huber = 0.0; // the robust cost's width, while `robust`

            // What a solve reads and may differ from one to the next; a pair
            // whose robust cost has been taken off does not get it back.
            std::vector<unsigned char> active;
            std::vector<unsigned char> robust;

            // Out, per pair: the squared error, weighted, of each of its two
            // observations.
            std::vector<double> chi2_1;
            std::vector<double> chi2_2;

            std::size_t size() const { return X1.size(); }

            void add(const Eigen::Vector3d &point1, const Eigen::Vector3d &point2, const Eigen::Vector2d &seen1,
                     const Eigen::Vector2d &seen2, double information1, double information2)
            {
                X1.push_back(point1);
                X2.push_back(point2);
                uv1.push_back(seen1);
                uv2.push_back(seen2);
                invSigma2_1.push_back(information1);
                invSigma2_2.push_back(information2);
                active.push_back(1);
                robust.push_back(1);
                chi2_1.push_back(0.0);
                chi2_2.push_back(0.0);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_SIM3PROBLEM_H
