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

#ifndef WELDINGBATASK_H
#define WELDINGBATASK_H

#include "optim/BaProblem.hpp"
#include "optim/BundleAdjuster.hpp"

#include <utility>
#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class MapPoint;

    // The bundle adjustment of the window where two maps were welded
    // (Optimizer::LocalBundleAdjustment with the keyframes given): the
    // keyframes to adjust, the ones to hold, and every point either sees.
    class WeldingBaTask
    {
    public:
        void Build(KeyFrame* pMainKF, const std::vector<KeyFrame*> &vpAdjustKF,
                   const std::vector<KeyFrame*> &vpFixedKF);

        // Five iterations with the robust cost; then, unless stopped, the
        // observations that do not fit are left out, the robust cost taken
        // off, and ten more.
        void Solve(optim::BundleAdjuster &solver, bool* pbStopFlag);

        // As LocalBaTask::Apply, under the update lock of pMainKF's map.
        void Apply(KeyFrame* pMainKF);

    private:
        // The observations Apply erases.
        const std::vector<std::pair<KeyFrame*, MapPoint*>> &Classify();

        optim::BaProblem mProblem;

        // In the order v1.0 walked them.
        std::vector<KeyFrame*> mvpAdjustKF;
        std::vector<int> mvnAdjustPose;
        std::vector<MapPoint*> mvpMarked; // every point of the window
        std::vector<MapPoint*> mvpPoints; // those that were not bad when the window was copied
        std::vector<int> mvnPoint;

        // By observation.
        std::vector<KeyFrame*> mvpObsKF;
        std::vector<MapPoint*> mvpObsMP;

        std::vector<std::pair<KeyFrame*, MapPoint*>> mvToErase;
    };

} // namespace ORB_SLAM3

#endif // WELDINGBATASK_H
