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

#ifndef LOCALBATASK_H
#define LOCALBATASK_H

#include "optim/BaProblem.hpp"
#include "optim/BundleAdjuster.hpp"

#include <utility>
#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    class MapPoint;

    // Optimizer::LocalBundleAdjustment of a keyframe in its three steps: find
    // the window and copy it, adjust the copy, write it back. The window is the
    // keyframes covisible with the keyframe, the points they see, and the other
    // keyframes that see those points, whose poses are held.
    class LocalBaTask
    {
    public:
        // False, and nothing to solve, when no keyframe of the window is held:
        // nothing would fix where the window is.
        bool Build(KeyFrame* pKF, Map* pMap);

        // Ten iterations at most; pbStopFlag ends them early.
        void Solve(optim::BundleAdjuster &solver, bool* pbStopFlag);

        // An observation that does not fit -- chi2 over 5.991 for two numbers,
        // 7.815 for three, or the point behind the camera -- is erased; then
        // the poses and the points are written. Takes the map's update lock.
        void Apply(Map* pMap);

        int FixedKeyFrames() const { return mnFixedKF; }
        int LocalKeyFrames() const { return static_cast<int>(mvnLocalPose.size()); }
        int Edges() const { return static_cast<int>(mProblem.observations()); }

        // For the build that runs v1.0's body beside this (Shadow.hpp): the
        // observations Apply would erase; whether they are `erased` and the
        // map holds the poses and points Apply would write; and the marks
        // Build left on the window taken off again, so that v1.0's body finds
        // the same window.
        const std::vector<std::pair<KeyFrame*, MapPoint*>> &Classify();
        bool Matches(const std::vector<std::pair<KeyFrame*, MapPoint*>> &erased) const;
        void ResetMarks();

    private:
        optim::BaProblem mProblem;
        bool mbInertial = false;
        bool mbDepth = false; // an observation with depth: stereo, or the second camera
        int mnFixedKF = 0;

        // In the order v1.0 walked them, which is the order they are written
        // back in.
        std::vector<KeyFrame*> mvpLocalKF;
        std::vector<int> mvnLocalPose; // the pose of each in the problem
        std::vector<KeyFrame*> mvpFixedKF;
        std::vector<MapPoint*> mvpPoints;
        std::vector<int> mvnPoint; // the point of each in the problem

        // By observation.
        std::vector<KeyFrame*> mvpObsKF;
        std::vector<MapPoint*> mvpObsMP;

        std::vector<std::pair<KeyFrame*, MapPoint*>> mvToErase;
    };

} // namespace ORB_SLAM3

#endif // LOCALBATASK_H
