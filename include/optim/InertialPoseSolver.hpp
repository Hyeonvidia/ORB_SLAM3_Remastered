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

#ifndef OPTIM_INERTIALPOSESOLVER_H
#define OPTIM_INERTIALPOSESOLVER_H

#include "optim/InertialPoseProblem.hpp"

#include <memory>
#include <vector>

namespace ORB_SLAM3
{

    namespace optim
    {

        // What solves an InertialPoseProblem, a round at a time as PoseSolver
        // does -- but a round goes on from where the one before it stopped.
        class InertialPoseSolver
        {
        public:
            virtual ~InertialPoseSolver() = default;

            // Takes the problem: its states are where the first Solve starts.
            // Every call that follows must be given that problem with nothing
            // changed but `active` and `robust`.
            virtual void Prepare(const InertialPoseProblem &problem) = 0;

            // Over the active observations and the inertial terms, at most
            // nIterations of Gauss-Newton. Leaves the states found in the
            // problem; the chi2 of an active observation as the last iteration
            // began with it, and of the others at the states found.
            virtual void Solve(InertialPoseProblem &problem, int nIterations) = 0;

            // The chi2 of every observation at the states found.
            virtual void Evaluate(InertialPoseProblem &problem) = 0;

            // Fills the problem's `information`, from the inertial terms and
            // the observations marked.
            virtual void Information(InertialPoseProblem &problem, const std::vector<unsigned char> &vbInlier) = 0;
        };

        // The solver of the library this was built with: src/optim_g2o/ today.
        std::unique_ptr<InertialPoseSolver> MakeInertialPoseSolver();

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_INERTIALPOSESOLVER_H
