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

#ifndef OPTIM_SIM3SOLVER_H
#define OPTIM_SIM3SOLVER_H

#include "optim/Sim3Problem.hpp"

#include <memory>

namespace ORB_SLAM3
{

    namespace optim
    {

        // What solves a Sim3Problem.
        class Sim3Solver
        {
        public:
            virtual ~Sim3Solver() = default;

            // Takes the problem: where S12 starts, and the pairs.
            virtual void Prepare(const Sim3Problem &problem) = 0;

            // One solve over the active pairs, the first from where Prepare was
            // told and each later one from where the one before ended. Leaves
            // S12 and, for the active pairs, the chi2 of their observations as
            // the solve left them.
            virtual void Solve(Sim3Problem &problem, int nIterations) = 0;

            // The chi2 of the active pairs at S12 as it now is.
            virtual void Evaluate(Sim3Problem &problem) = 0;
        };

        // The solver of the library this was built with: src/optim_g2o/ today.
        std::unique_ptr<Sim3Solver> MakeSim3Solver();

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_SIM3SOLVER_H
