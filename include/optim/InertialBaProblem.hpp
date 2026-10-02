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

#ifndef OPTIM_INERTIALBAPROBLEM_H
#define OPTIM_INERTIALBAPROBLEM_H

#include "optim/InertialState.hpp"
#include "optim/Rig.hpp"

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

        // A bundle adjustment with an inertial unit: states and points, where
        // each point was seen from each state, and what the unit measured
        // between pairs of states. What Optimizer::LocalInertialBA,
        // FullInertialBA and MergeInertialBA solve.
        //
        // The unknowns are laid out in the order of the arrays -- the poses,
        // then velocity and biases state by state, then the biases all states
        // share if they do, then the points -- and the terms are summed in the
        // order of theirs: the inertial terms, each followed by the walks of
        // its two biases; the priors on shared biases; the observations.
        struct InertialBaProblem
        {
            // In: where to start. Out: what was found. A state is held whole or
            // not at all; one that is not `inertial` is a pose alone.
            std::vector<InertialState> states;
            std::vector<unsigned char> fixed;
            std::vector<unsigned char> inertial;

            // One pair of biases for every state in place of each its own, as
            // when the unit is first tied to a map. Then there are no walks,
            // and the pair may be held towards zero, accelerometer first, the
            // information of each this times identity.
            bool sharedBias = false;
            Eigen::Vector3d sharedGyroBias = Eigen::Vector3d::Zero();
            Eigen::Vector3d sharedAccBias = Eigen::Vector3d::Zero();
            bool biasPriors = false;
            double accPriorInformation = 0.0;
            double gyroPriorInformation = 0.0;

            // In and out likewise.
            std::vector<Eigen::Vector3d> Xw;

            // One entry per inertial term: what was measured between two
            // states, integrated. The preintegrations are not the problem's:
            // whoever builds it keeps them for as long as it is solved.
            std::vector<int> from;
            std::vector<int> to;
            std::vector<IMU::Preintegrated*> preintegration;
            std::vector<double> inertialHuber; // the robust cost's width; 0 for none
            std::vector<double> inertialScale; // its information is the preintegration's times this
            std::vector<Eigen::Matrix3d> gyroWalkInformation;
            std::vector<Eigen::Matrix3d> accWalkInformation;

            // One entry per observation. The cameras are those of the state's
            // pose: kMono in the first, kStereo in the first with its baseline,
            // kRight in the second.
            std::vector<unsigned char> kind; // ObservationKind
            std::vector<int> state;
            std::vector<int> point;
            std::vector<Eigen::Vector3d> uv; // (u, v, uR); uR unused but for kStereo
            std::vector<double> invSigma2;   // the information of each is this times identity
            std::vector<double> huber;       // the robust cost's width

            // Out, per observation: its squared error, weighted, as the solve
            // left it, and whether the point is in front of the camera.
            std::vector<double> chi2;
            std::vector<unsigned char> depthPositive;

            // Out: the robust cost of all terms before the first iteration and
            // as the last one left it.
            double costBefore = 0.0;
            double costAfter = 0.0;

            std::size_t terms() const { return from.size(); }
            std::size_t observations() const { return kind.size(); }

            int addState(const InertialState &s, bool bFixed, bool bInertial)
            {
                states.push_back(s);
                fixed.push_back(bFixed);
                inertial.push_back(bInertial);
                return static_cast<int>(states.size()) - 1;
            }

            int addPoint(const Eigen::Vector3d &X)
            {
                Xw.push_back(X);
                return static_cast<int>(Xw.size()) - 1;
            }

            void addTerm(int i, int j, IMU::Preintegrated* pInt, double width, double scale,
                         const Eigen::Matrix3d &gyroWalk, const Eigen::Matrix3d &accWalk)
            {
                from.push_back(i);
                to.push_back(j);
                preintegration.push_back(pInt);
                inertialHuber.push_back(width);
                inertialScale.push_back(scale);
                gyroWalkInformation.push_back(gyroWalk);
                accWalkInformation.push_back(accWalk);
            }

            void addObservation(ObservationKind k, int iState, int iPoint, const Eigen::Vector3d &observed,
                                double information, double width)
            {
                kind.push_back(k);
                state.push_back(iState);
                point.push_back(iPoint);
                uv.push_back(observed);
                invSigma2.push_back(information);
                huber.push_back(width);
                chi2.push_back(0.0);
                depthPositive.push_back(1);
            }
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_INERTIALBAPROBLEM_H
