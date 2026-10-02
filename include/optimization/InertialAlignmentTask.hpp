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

#ifndef INERTIALALIGNMENTTASK_H
#define INERTIALALIGNMENTTASK_H

#include "common/ImuTypes.hpp"
#include "optim/InertialAlignmentProblem.hpp"
#include "optim/InertialAlignmentSolver.hpp"
#include "optim/SolveOptions.hpp"
#include "optimization/Digest.hpp"

#include <Eigen/Core>

#include <deque>
#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;

    // The three Optimizer::InertialOptimization in three steps. They are one
    // optimisation over every keyframe of a map with its poses held, and
    // differ in what is found:
    //
    //   when the inertial unit is first tied to the map   gravity, the biases
    //       and the velocities, and the scale if the map is of one camera
    //   with gravity and scale known                      the biases and the
    //       velocities
    //   to refine gravity and scale                       those two alone
    class InertialAlignmentTask
    {
    public:
        // Each reads every keyframe of the map and what was measured between
        // it and the one before, and says what is held and how to solve. In
        // the first two the keyframes' preintegrations first take the bias of
        // the keyframe before, as v1.0 gave it to them where the biases were
        // to be found.
        void Build(Map* pMap, const Eigen::Matrix3d &Rwg, double scale, bool bMono, bool bFixedVel, float priorG,
                   float priorA);
        void Build(Map* pMap, float priorG, float priorA);
        void Build(Map* pMap, const Eigen::Matrix3d &Rwg, double scale);

        void Solve(optim::InertialAlignmentSolver &solver);

        // What was found.
        const optim::InertialAlignmentProblem &Problem() const { return mProblem; }

        // Every keyframe takes its velocity and the biases found; one whose
        // gyroscope bias moved by more than 0.01 integrates its measurements
        // again.
        void Apply() const;

        // For the build that runs v1.0's body beside this (Shadow.hpp): what
        // Build read, and whether the keyframes hold what Apply would write.
        Digest Input() const;
        bool Holds() const;

    private:
        void Read(Map* pMap, bool bSetBias);

        optim::InertialAlignmentProblem mProblem;
        optim::SolveOptions mOptions;
        std::deque<IMU::Preintegrated> mPreintegrations; // copies: the problem points at them
        std::vector<KeyFrame*> mvpKFs;                   // the map's, as listed
        std::vector<int> mvnPose;                        // by keyframe as listed: its pose in the problem, or -1
        std::vector<KeyFrame*> mvpPoseKF;                // by pose of the problem
    };

} // namespace ORB_SLAM3

#endif // INERTIALALIGNMENTTASK_H
