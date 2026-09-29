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

#ifndef WORKERPOOL_H
#define WORKERPOOL_H

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ORB_SLAM3
{

    // A few threads that wait for one caller. Run(n, f) calls f(0) ... f(n - 1),
    // each once, on the workers and on the calling thread, and returns when
    // every call has returned. Items are taken in order, so the longest goes
    // first. What f throws on a worker is thrown again by Run.
    //
    // One caller at a time: what runs things in parallel owns its pool. The
    // workers sleep between calls, and one that is never needed costs a stack.
    class WorkerPool
    {
    public:
        // nThreads counts the caller: 1 starts no thread and Run is a loop.
        explicit WorkerPool(int nThreads);
        ~WorkerPool();

        WorkerPool(const WorkerPool &) = delete;
        WorkerPool &operator=(const WorkerPool &) = delete;

        void Run(int n, const std::function<void(int)> &f);

        int Threads() const { return static_cast<int>(mvWorkers.size()) + 1; }

    private:
        void Work();

        std::vector<std::thread> mvWorkers;
        std::mutex mMutex;
        std::condition_variable mWake;
        std::condition_variable mDone;
        const std::function<void(int)>* mpJob = nullptr;
        int mnItems = 0;
        int mnNext = 0;
        int mnRunning = 0;
        std::exception_ptr mpThrown;
        bool mbStop = false;
    };

} // namespace ORB_SLAM3

#endif // WORKERPOOL_H
