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

#ifndef OPTIM_INERTIALALIGNMENTSOLVER_H
#define OPTIM_INERTIALALIGNMENTSOLVER_H

#include "optim/InertialAlignmentProblem.hpp"
#include "optim/SolveOptions.hpp"

#include <memory>

namespace ORB_SLAM3
{

    namespace optim
    {

        // What solves an InertialAlignmentProblem, in one go: what was found is
        // left in the problem.
        class InertialAlignmentSolver
        {
        public:
            virtual ~InertialAlignmentSolver() = default;

            virtual void Solve(InertialAlignmentProblem &problem, const SolveOptions &options) = 0;
        };

        // The solver of the library this was built with: src/optim_g2o/ today.
        std::unique_ptr<InertialAlignmentSolver> MakeInertialAlignmentSolver();

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_INERTIALALIGNMENTSOLVER_H
