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

// See optimization/Shadow.hpp. Nothing of this file is in a build without
// ORBSLAM3R_OPT_SHADOW.
#ifdef ORBSLAM3R_OPT_SHADOW

#include "optimization/Shadow.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace ORB_SLAM3
{

    namespace shadow
    {

        namespace
        {
            struct Tally
            {
                const char* task = nullptr;
                std::atomic<long> calls{0};
                std::atomic<long> differ{0};
                std::atomic<long> moved{0};
            };

            // A fixed table, written from several threads and printed once.
            struct Tallies
            {
                Tally rows[32];
                std::mutex mutex;

                Tally &Row(const char* task)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    for(Tally &row : rows)
                    {
                        if(!row.task)
                            row.task = task;
                        if(std::strcmp(row.task, task) == 0)
                            return row;
                    }
                    return rows[31];
                }

                ~Tallies()
                {
                    for(const Tally &row : rows)
                        if(row.task)
                        {
                            std::fprintf(stderr, "OPT_SHADOW %s: %ld calls, %ld differ", row.task, row.calls.load(),
                                         row.differ.load());
                            if(row.moved.load())
                                std::fprintf(stderr, ", %ld not compared (the map moved between the two)",
                                             row.moved.load());
                            std::fprintf(stderr, "\n");
                        }
                }
            };

            Tallies gTallies;
        } // namespace

        void Count(const char* task, bool same)
        {
            Tally &row = gTallies.Row(task);
            row.calls++;
            if(!same)
            {
                if(row.differ++ < 5)
                    std::fprintf(stderr, "OPT_SHADOW %s: call %ld differs\n", task, row.calls.load());
            }
        }

        void Moved(const char* task)
        {
            Tally &row = gTallies.Row(task);
            row.calls++;
            row.moved++;
        }

    } // namespace shadow

} // namespace ORB_SLAM3

#endif // ORBSLAM3R_OPT_SHADOW
