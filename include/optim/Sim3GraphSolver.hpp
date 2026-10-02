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

#ifndef OPTIM_SIM3GRAPHSOLVER_H
#define OPTIM_SIM3GRAPHSOLVER_H

#include "optim/Sim3GraphProblem.hpp"
#include "optim/SolveOptions.hpp"

#include <memory>

namespace ORB_SLAM3
{

    namespace optim
    {

        // What solves a Sim3GraphProblem, in one go: the poses found are left
        // in the problem.
        class Sim3GraphSolver
        {
        public:
            virtual ~Sim3GraphSolver() = default;

            virtual void Solve(Sim3GraphProblem &problem, const SolveOptions &options) = 0;
        };

        // The solver of the library this was built with: src/optim_g2o/ today.
        std::unique_ptr<Sim3GraphSolver> MakeSim3GraphSolver();

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_SIM3GRAPHSOLVER_H
