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

#ifndef INERTIALPOSETASK_H
#define INERTIALPOSETASK_H

#include "common/ImuTypes.hpp"
#include "optim/InertialPoseProblem.hpp"
#include "optim/InertialPoseSolver.hpp"

#include <Eigen/Core>

#include <vector>

namespace ORB_SLAM3
{

    class Frame;
    class MapPoint;

    // Optimizer::PoseInertialOptimizationLastKeyFrame and ...LastFrame in three
    // steps. The two are one optimisation of the frame's pose, velocity and
    // biases, from the points it has matched and from what the inertial unit
    // measured since an earlier state; they differ in which state that is. The
    // last keyframe is held where it is. The frame before is not: it moves
    // with the frame, tied to what the optimisation before this one left known
    // of it, and is then summed out of what this one leaves known of the frame.
    class InertialPoseTask
    {
    public:
        enum Previous
        {
            kLastKeyFrame,
            kLastFrame,
        };

        // The caller holds MapPoint::mGlobalMutex: the points are read here.
        void Build(Frame* pFrame, Previous previous);

        // Four rounds of ten iterations, each from where the one before
        // stopped. After each an observation is in or out by its chi2, a point
        // nearer than ten metres being allowed half as much again; the fourth
        // round is without the robust cost. With fewer than thirty left in,
        // and unless bRecInit, those that are merely not far out are taken
        // back.
        void Solve(optim::InertialPoseSolver &solver, bool bRecInit);

        // The state found, which matches did not fit, and what is now known of
        // the frame's state for the next optimisation to start from; the frame
        // before gives up what was known of its own. Returns how many fitted.
        int Apply(Frame* pFrame) const;

    private:
        void Add(optim::ObservationKind kind, int nFeature, MapPoint* pMP, const Eigen::Vector3d &obs, float invSigma2,
                 float thHuber);

        optim::InertialPoseProblem mProblem;
        IMU::Preintegrated mPreintegration; // a copy: the problem points at it
        Previous mPrevious = kLastKeyFrame;
        std::vector<int> mvnFeature;           // which feature of the frame each observation is
        std::vector<unsigned char> mvbClose;   // by observation: its point is nearer than ten metres
        std::vector<unsigned char> mvbOutlier; // by observation
        int mnBad = 0;
    };

} // namespace ORB_SLAM3

#endif // INERTIALPOSETASK_H
