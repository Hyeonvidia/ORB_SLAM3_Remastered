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

#ifndef OPTIM_POSEPROBLEM_H
#define OPTIM_POSEPROBLEM_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ORB_SLAM3
{

    class GeometricCamera;

    namespace optim
    {

        // The pose of one frame from points whose positions are taken as known:
        // what Optimizer::PoseOptimization solves, said without the frame, the
        // map or the library that solves it. Whoever builds it copies what it
        // needs out of the frame under the map's lock; whoever solves it sees
        // nothing else.
        //
        // The pose is camera-from-world and what is observed is in pixels, as
        // the system has them.
        struct PoseProblem
        {
            enum Kind : std::uint8_t
            {
                kMono,   // (u, v) in `camera`
                kStereo, // (u, v, uR) of a rectified pair: fx, fy, cx, cy, bf
                kRight,  // (u, v) in `camera2`, which is right-from-left of `camera`
            };

            // In: where to start. Out: what was found.
            Eigen::Quaterniond Rcw = Eigen::Quaterniond::Identity();
            Eigen::Vector3d tcw = Eigen::Vector3d::Zero();

            GeometricCamera* camera = nullptr;
            GeometricCamera* camera2 = nullptr;
            Eigen::Quaterniond Rrl = Eigen::Quaterniond::Identity();
            Eigen::Vector3d trl = Eigen::Vector3d::Zero();
            double fx = 0, fy = 0, cx = 0, cy = 0, bf = 0;

            // One entry per observation, in the order they are to be summed.
            std::vector<std::uint8_t> kind;
            std::vector<Eigen::Vector3d> Xw;
            std::vector<Eigen::Vector3d> uv; // (u, v, uR); uR unused but for kStereo
            std::vector<double> invSigma2;   // the information of each is this times identity
            std::vector<double> huber;       // the robust cost's width, while `robust`

            // What a solve reads besides the pose and may differ from one solve
            // to the next: whether the observation takes part, and whether its
            // cost is the robust one.
            std::vector<std::uint8_t> active;
            std::vector<std::uint8_t> robust;

            // Out, for every observation whether active or not: its squared
            // error, weighted, at the pose found.
            std::vector<double> chi2;

            std::size_t size() const { return kind.size(); }

            void reserve(std::size_t n)
            {
                kind.reserve(n);
                Xw.reserve(n);
                uv.reserve(n);
                invSigma2.reserve(n);
                huber.reserve(n);
                active.reserve(n);
                robust.reserve(n);
                chi2.reserve(n);
            }

            void add(Kind k, const Eigen::Vector3d &point, const Eigen::Vector3d &observed, double information,
                     double width)
            {
                kind.push_back(k);
                Xw.push_back(point);
                uv.push_back(observed);
                invSigma2.push_back(information);
                huber.push_back(width);
                active.push_back(1);
                robust.push_back(1);
                chi2.push_back(0.0);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_POSEPROBLEM_H
