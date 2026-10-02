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

#ifndef OPTIM_BUNDLEADJUSTER_H
#define OPTIM_BUNDLEADJUSTER_H

#include "optim/BaProblem.hpp"
#include "optim/SolveOptions.hpp"

#include <memory>

namespace ORB_SLAM3
{

    namespace optim
    {

        // What solves a BaProblem: Levenberg-Marquardt over the poses that are
        // not fixed and the points, the points eliminated first.
        class BundleAdjuster
        {
        public:
            virtual ~BundleAdjuster() = default;

            // Takes the problem: its poses, points and observations.
            virtual void Prepare(const BaProblem &problem) = 0;

            // One solve over the active observations. The first starts from the
            // poses and points Prepare was given and each later one from where
            // the one before ended; of the problem it reads `active` and
            // `robust`, and it leaves in it the poses and points found and the
            // chi2 and depth sign of every observation.
            virtual void Solve(BaProblem &problem, const SolveOptions &options) = 0;
        };

        // The solver of the library this was built with: src/optim_g2o/ today.
        std::unique_ptr<BundleAdjuster> MakeBundleAdjuster();

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_BUNDLEADJUSTER_H
