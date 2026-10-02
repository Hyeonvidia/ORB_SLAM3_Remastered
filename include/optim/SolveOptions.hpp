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

#ifndef OPTIM_SOLVEOPTIONS_H
#define OPTIM_SOLVEOPTIONS_H

namespace ORB_SLAM3
{

    namespace optim
    {

        // How one solve is to be run. What is not set is the solver's own.
        struct SolveOptions
        {
            int nIterations = 10;

            // Read by the solver between its steps; when it becomes true the
            // solve ends where it is. Written by another thread.
            bool* pbStop = nullptr;

            // Levenberg-Marquardt's damping at the first iteration: a value,
            // or a fraction of the largest diagonal entry of the Hessian.
            enum Damping
            {
                kSolverDefault,
                kValue,
                kFractionOfDiagonal,
            };
            Damping damping = kSolverDefault;
            double dampingValue = 0.0;

            // End after this many iterations in a row that each gained less
            // than a thousandth. 0: the solver's own.
            int nStallIterations = 0;
        };

    } // namespace optim

} // namespace ORB_SLAM3

#endif // OPTIM_SOLVEOPTIONS_H
