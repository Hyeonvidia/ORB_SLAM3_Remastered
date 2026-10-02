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

#ifndef GLOBALBATASK_H
#define GLOBALBATASK_H

#include "optim/BaProblem.hpp"
#include "optim/BundleAdjuster.hpp"

#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    class MapPoint;

    // Optimizer::BundleAdjustment over the keyframes and points given -- all
    // of a map, after a loop or at a monocular initialisation -- in three
    // steps. Only the map's first keyframe is held.
    class GlobalBaTask
    {
    public:
        // bRobust: the observations of the first camera and of a rectified
        // pair get the robust cost; those of a second camera always do.
        void Build(const std::vector<KeyFrame*> &vpKFs, const std::vector<MapPoint*> &vpMP, bool bRobust);

        void Solve(optim::BundleAdjuster &solver, int nIterations, bool* pbStopFlag);

        // To the keyframes and points themselves when nLoopKF is the map's
        // origin keyframe; otherwise beside them (mTcwGBA, mPosGBA), for Loop
        // Closing to apply once Local Mapping has stopped.
        void Apply(unsigned long nLoopKF) const;

    private:
        optim::BaProblem mProblem;
        Map* mpMap = nullptr;

        // As given.
        std::vector<KeyFrame*> mvpKF;
        std::vector<int> mvnPose; // the pose of each in the problem, or -1
        std::vector<MapPoint*> mvpMP;
        std::vector<int> mvnPoint; // the point of each in the problem, or -1

        // By pose and by point of the problem.
        std::vector<KeyFrame*> mvpPoseKF;
        std::vector<MapPoint*> mvpPointMP;
    };

} // namespace ORB_SLAM3

#endif // GLOBALBATASK_H
