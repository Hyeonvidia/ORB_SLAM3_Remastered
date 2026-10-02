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

#ifndef OPTIMIZATION_SHADOW_H
#define OPTIMIZATION_SHADOW_H

// A build with -DORBSLAM3R_OPT_SHADOW=ON runs every optimisation that has been
// taken apart twice on the same input -- as it is now, and as v1.0 wrote it --
// and counts the calls whose results differ in any bit. The counts are printed
// when the process ends. It is how taking an optimisation apart is shown to
// have changed nothing; the build is not for measuring time.

#ifdef ORBSLAM3R_OPT_SHADOW

#include <Eigen/Core>
#include <g2o/types/sim3/sim3.h>

#include "optimization/KeyFrameAndPose.hpp"

#include <map>
#include <set>
#include <vector>

namespace ORB_SLAM3
{

    class Frame;
    class KeyFrame;
    class Map;
    class MapPoint;

    namespace shadow
    {

        int PoseOptimization(Frame* pFrame);
        void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, int &num_fixedKF, int &num_OptKF,
                                   int &num_MPs, int &num_edges);
        void WeldingBundleAdjustment(KeyFrame* pMainKF, std::vector<KeyFrame*> vpAdjustKF,
                                     std::vector<KeyFrame*> vpFixedKF, bool* pbStopFlag);

        void BundleAdjustment(const std::vector<KeyFrame*> &vpKFs, const std::vector<MapPoint*> &vpMP, int nIterations,
                              bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust);

        int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches1, g2o::Sim3 &g2oS12,
                         const float th2, const bool bFixScale, Eigen::Matrix<double, 7, 7> &mAcumHessian,
                         const bool bAllPoints);

        void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                    const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                    const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections,
                                    const bool &bFixScale);
        void OptimizeEssentialGraph(KeyFrame* pCurKF, std::vector<KeyFrame*> &vpFixedKFs,
                                    std::vector<KeyFrame*> &vpFixedCorrectedKFs, std::vector<KeyFrame*> &vpNonFixedKFs,
                                    std::vector<MapPoint*> &vpNonCorrectedMPs);
        void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                        const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                        const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections);

        // One line per task at exit: calls, and calls that differed. Moved
        // counts a call that could not be compared: a global adjustment runs
        // while the other threads go on changing the map, and when the two
        // did not read the same (optimization/Digest.hpp) their results say
        // nothing about each other.
        void Count(const char* task, bool same);
        void Moved(const char* task);

    } // namespace shadow

} // namespace ORB_SLAM3

#endif // ORBSLAM3R_OPT_SHADOW

#endif // OPTIMIZATION_SHADOW_H
