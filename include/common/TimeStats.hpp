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

#ifndef TIMESTATS_H
#define TIMESTATS_H

#include <cmath>
#include <vector>

namespace ORB_SLAM3
{

    // The mean and deviation that the REGISTER_TIMES report prints.
    //
    // They were file-local functions of Tracking.cpp, because Tracking printed
    // the whole report -- its own timings and, by reaching into their public
    // members, Local Mapping's and Loop Closing's. Each thread now prints what
    // it measured, so the three of them share these.
    //
    // The counts skip zeros and the timings do not; an empty series gives NaN.
    // Both are what v1.0 prints, and the report is compared against it.
    namespace TimeStats
    {

        inline double Average(const std::vector<double> &vTimes)
        {
            double accum = 0;
            for(double value : vTimes)
            {
                accum += value;
            }

            return accum / vTimes.size();
        }

        inline double Deviation(const std::vector<double> &vTimes, double average)
        {
            double accum = 0;
            for(double value : vTimes)
            {
                accum += std::pow(value - average, 2);
            }
            return std::sqrt(accum / vTimes.size());
        }

        inline double Average(const std::vector<int> &vValues)
        {
            double accum = 0;
            int total = 0;
            for(double value : vValues)
            {
                if(value == 0)
                    continue;
                accum += value;
                total++;
            }

            return accum / total;
        }

        inline double Deviation(const std::vector<int> &vValues, double average)
        {
            double accum = 0;
            int total = 0;
            for(double value : vValues)
            {
                if(value == 0)
                    continue;
                accum += std::pow(value - average, 2);
                total++;
            }
            return std::sqrt(accum / total);
        }

    } // namespace TimeStats

} // namespace ORB_SLAM3

#endif // TIMESTATS_H
