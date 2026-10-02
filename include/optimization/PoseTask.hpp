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

#ifndef POSETASK_H
#define POSETASK_H

#include "optim/PoseProblem.hpp"
#include "optim/PoseSolver.hpp"

#include <vector>

namespace ORB_SLAM3
{

    class Frame;

    // Optimizer::PoseOptimization in the three steps it always was -- copy
    // what the frame has matched, find the pose in rounds, write the pose and
    // which matches did not fit back -- with the middle one seeing neither the
    // frame nor the map.
    class PoseTask
    {
    public:
        // The caller holds MapPoint::mGlobalMutex: the points are read here.
        void Build(Frame* pFrame);

        // Four rounds of ten iterations, each from the frame's pose. After each
        // an observation is in or out by its chi2 at the pose found -- 5.991
        // for two degrees of freedom, 7.815 for three -- and out means out of
        // the next round only; the fourth is without the robust cost.
        void Solve(optim::PoseSolver &solver);

        // The pose found and, for each observation, whether it was left out of
        // the last round. Returns how many were not.
        int Apply(Frame* pFrame) const;

    private:
        optim::PoseProblem mProblem;
        std::vector<int> mvnFeature;           // which feature of the frame each observation is
        std::vector<unsigned char> mvbOutlier; // by observation
        int mnBad = 0;
        bool mbSolved = false;
    };

} // namespace ORB_SLAM3

#endif // POSETASK_H
