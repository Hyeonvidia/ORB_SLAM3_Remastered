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

#include <vector>

namespace ORB_SLAM3
{

    class Frame;
    class KeyFrame;
    class Map;

    namespace shadow
    {

        int PoseOptimization(Frame* pFrame);
        void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, int &num_fixedKF, int &num_OptKF,
                                   int &num_MPs, int &num_edges);
        void WeldingBundleAdjustment(KeyFrame* pMainKF, std::vector<KeyFrame*> vpAdjustKF,
                                     std::vector<KeyFrame*> vpFixedKF, bool* pbStopFlag);

        // One line per task at exit: calls, and calls that differed.
        void Count(const char* task, bool same);

    } // namespace shadow

} // namespace ORB_SLAM3

#endif // ORBSLAM3R_OPT_SHADOW

#endif // OPTIMIZATION_SHADOW_H
