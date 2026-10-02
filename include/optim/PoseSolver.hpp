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

#ifndef OPTIM_POSESOLVER_H
#define OPTIM_POSESOLVER_H

#include "optim/PoseProblem.hpp"

#include <memory>

namespace ORB_SLAM3
{

    namespace optim
    {

        // What solves a PoseProblem. The rounds of an optimisation -- solve,
        // see which observations fit, leave the others out, solve again -- are
        // the caller's; this does one of them at a time over the same
        // observations.
        class PoseSolver
        {
        public:
            virtual ~PoseSolver() = default;

            // Takes the observations of the problem. Every Solve that follows
            // must be given that problem with nothing changed but the pose,
            // `active` and `robust`.
            virtual void Prepare(const PoseProblem &problem) = 0;

            // From the problem's pose, over its active observations, at most
            // nIterations of Levenberg-Marquardt. Leaves the pose found and the
            // chi2 of every observation in the problem.
            virtual void Solve(PoseProblem &problem, int nIterations) = 0;
        };

        // The solver of the library this was built with: src/optim_g2o/ today.
        std::unique_ptr<PoseSolver> MakePoseSolver();

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_POSESOLVER_H
