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

#ifndef SIM3TASK_H
#define SIM3TASK_H

#include "optim/Sim3Problem.hpp"
#include "optim/Sim3Solver.hpp"

#include "common/Sim3.hpp"

#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class MapPoint;

    // Optimizer::OptimizeSim3 in three steps: the pairs of matched points of
    // two keyframes copied, the similarity between the keyframes found from
    // them, and the matches that did not fit taken off the list.
    class Sim3Task
    {
    public:
        // vpMatches1[i] is the point of pKF2 matched to feature i of pKF1.
        void Build(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatches1, const Sim3 &g2oS12,
                   float th2, bool bFixScale, bool bAllPoints);

        // Five iterations; the pairs with an observation over th2 left out and
        // the robust cost taken off the rest; then five more, or ten if any
        // was left out -- unless fewer than ten pairs remain, and then nothing
        // more.
        void Solve(optim::Sim3Solver &solver);

        // The matches left out are set to null. Returns the number that fit
        // and gives the similarity found, or 0 and no similarity when fewer
        // than ten pairs were left after the first round.
        int Apply(std::vector<MapPoint*> &vpMatches1, Sim3 &g2oS12, Eigen::Matrix<double, 7, 7> &mAcumHessian) const;

    private:
        optim::Sim3Problem mProblem;
        float mTh2 = 0.f;
        std::vector<int> mvnMatch;         // which match each pair is
        std::vector<unsigned char> mvbOut; // by pair: left out, in either round
        bool mbFound = false;
        int mnIn = 0;
    };

} // namespace ORB_SLAM3

#endif // SIM3TASK_H
