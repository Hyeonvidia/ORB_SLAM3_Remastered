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

#ifndef OPTIM_INERTIALPOSEPROBLEM_H
#define OPTIM_INERTIALPOSEPROBLEM_H

#include "optim/InertialState.hpp"
#include "optim/Rig.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ORB_SLAM3
{

    namespace IMU
    {
        class Preintegrated;
    }

    namespace optim
    {

        // The state of one frame from points whose positions are taken as known
        // and from what the inertial unit measured since an earlier state: what
        // Optimizer::PoseInertialOptimizationLastKeyFrame and ...LastFrame
        // solve. The earlier state is either held, or free and tied to what was
        // known of it before.
        //
        // The cameras an observation is projected with are those of
        // `state.pose`: kMono in the first, kStereo in the first with its
        // baseline, kRight in the second.
        struct InertialPoseProblem
        {
            // In: where to start. Out: what was found.
            InertialState state;
            InertialState previous;
            bool previousFixed = true;

            // What was measured between the two, integrated. Not the problem's:
            // whoever builds the problem keeps it for as long as it is solved.
            IMU::Preintegrated* preintegration = nullptr;

            // How far each bias may have walked between the two.
            Eigen::Matrix3d gyroWalkInformation = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d accWalkInformation = Eigen::Matrix3d::Identity();

            // What was known of the earlier state, when it is not held: where
            // it was thought to be, and how firmly, in the order rotation,
            // position, velocity, gyroscope bias, accelerometer bias.
            Eigen::Matrix3d priorRwb = Eigen::Matrix3d::Identity();
            Eigen::Vector3d priorTwb = Eigen::Vector3d::Zero();
            Eigen::Vector3d priorVelocity = Eigen::Vector3d::Zero();
            Eigen::Vector3d priorGyroBias = Eigen::Vector3d::Zero();
            Eigen::Vector3d priorAccBias = Eigen::Vector3d::Zero();
            Eigen::Matrix<double, 15, 15> priorInformation = Eigen::Matrix<double, 15, 15>::Identity();
            double priorHuber = 5.0;

            // One entry per observation, in the order they are to be summed.
            std::vector<std::uint8_t> kind; // ObservationKind
            std::vector<Eigen::Vector3d> Xw;
            std::vector<Eigen::Vector3d> uv; // (u, v, uR); uR unused but for kStereo
            std::vector<double> invSigma2;   // the information of each is this times identity
            std::vector<double> huber;       // the robust cost's width, while `robust`

            // What a solve reads besides and may differ from one solve to the
            // next: whether the observation takes part, and whether its cost is
            // the robust one.
            std::vector<std::uint8_t> active;
            std::vector<std::uint8_t> robust;

            // Out, for every observation whether active or not: its squared
            // error, weighted, and whether the point is in front of the camera.
            std::vector<double> chi2;
            std::vector<std::uint8_t> depthPositive;

            // Out, when asked for: how firmly the solution holds the states,
            // J'WJ of every term without its robust weight. 15x15 over the
            // frame's state when the earlier one is held; 30x30 over the
            // earlier state and then the frame's when it is not. A state is in
            // the order of the prior above.
            Eigen::MatrixXd information;

            std::size_t size() const { return kind.size(); }

            void add(ObservationKind k, const Eigen::Vector3d &point, const Eigen::Vector3d &observed,
                     double information_, double width)
            {
                kind.push_back(k);
                Xw.push_back(point);
                uv.push_back(observed);
                invSigma2.push_back(information_);
                huber.push_back(width);
                active.push_back(1);
                robust.push_back(1);
                chi2.push_back(0.0);
                depthPositive.push_back(1);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_INERTIALPOSEPROBLEM_H
