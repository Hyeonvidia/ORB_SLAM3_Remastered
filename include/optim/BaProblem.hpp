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

#ifndef OPTIM_BAPROBLEM_H
#define OPTIM_BAPROBLEM_H

#include "optim/Rig.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <vector>

namespace ORB_SLAM3
{

    namespace optim
    {

        // A bundle adjustment: poses and points, and where each point was seen
        // from each pose. What the local adjustment of a keyframe's window, the
        // one of a merge's welding window and the global one solve, said
        // without keyframes, map points or the library that solves it.
        //
        // Poses are camera-from-world. The unknowns are laid out in the order
        // of the arrays -- the poses that are not fixed, then the points -- and
        // the observations are summed in the order of theirs; a solver that
        // keeps to both gives the same bits for the same problem.
        struct BaProblem
        {
            // In: where to start. Out: what was found.
            std::vector<Eigen::Quaterniond> Rcw;
            std::vector<Eigen::Vector3d> tcw;
            std::vector<unsigned char> poseFixed;
            std::vector<int> poseRig; // into `rigs`
            std::vector<Rig> rigs;

            // In and out likewise. A point no observation names is left as it is.
            std::vector<Eigen::Vector3d> Xw;

            // One entry per observation.
            std::vector<unsigned char> kind; // ObservationKind
            std::vector<int> pose;
            std::vector<int> point;
            std::vector<Eigen::Vector3d> uv; // (u, v, uR); uR unused but for kStereo
            std::vector<double> invSigma2;   // the information of each is this times identity
            std::vector<double> huber;       // the robust cost's width, while `robust`

            // What a solve reads and may differ from one solve to the next:
            // whether the observation takes part, and whether its cost is the
            // robust one. An observation whose robust cost has been taken off
            // does not get it back.
            std::vector<unsigned char> active;
            std::vector<unsigned char> robust;

            // Out, per observation: its squared error, weighted, as the solve
            // left it, and whether the point is in front of the camera.
            std::vector<double> chi2;
            std::vector<unsigned char> depthPositive;

            std::size_t poses() const { return Rcw.size(); }
            std::size_t points() const { return Xw.size(); }
            std::size_t observations() const { return kind.size(); }

            // The rig's place in `rigs`; the same rig twice in a row is one.
            int addRig(const Rig &rig)
            {
                if(rigs.empty() || !(rigs.back() == rig))
                    rigs.push_back(rig);
                return static_cast<int>(rigs.size()) - 1;
            }

            int addPose(const Eigen::Quaterniond &R, const Eigen::Vector3d &t, bool fixed, int rig)
            {
                Rcw.push_back(R);
                tcw.push_back(t);
                poseFixed.push_back(fixed);
                poseRig.push_back(rig);
                return static_cast<int>(Rcw.size()) - 1;
            }

            int addPoint(const Eigen::Vector3d &X)
            {
                Xw.push_back(X);
                return static_cast<int>(Xw.size()) - 1;
            }

            void addObservation(ObservationKind k, int iPose, int iPoint, const Eigen::Vector3d &observed,
                                double information, double width, bool bRobust = true)
            {
                kind.push_back(k);
                pose.push_back(iPose);
                point.push_back(iPoint);
                uv.push_back(observed);
                invSigma2.push_back(information);
                huber.push_back(width);
                active.push_back(1);
                robust.push_back(bRobust);
                chi2.push_back(0.0);
                depthPositive.push_back(1);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_BAPROBLEM_H
