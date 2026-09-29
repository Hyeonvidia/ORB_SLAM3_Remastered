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

#include "common/WorkerPool.hpp"

namespace ORB_SLAM3
{

    WorkerPool::WorkerPool(int nThreads)
    {
        for(int i = 1; i < nThreads; ++i)
            mvWorkers.emplace_back(&WorkerPool::Work, this);
    }

    WorkerPool::~WorkerPool()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mbStop = true;
        }
        mWake.notify_all();
        for(std::thread &worker : mvWorkers)
            worker.join();
    }

    void WorkerPool::Run(int n, const std::function<void(int)> &f)
    {
        if(mvWorkers.empty() || n <= 1)
        {
            for(int i = 0; i < n; ++i)
                f(i);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mMutex);
            mpJob = &f;
            mnItems = n;
            mnNext = 0;
            mpThrown = nullptr;
        }
        mWake.notify_all();

        std::unique_lock<std::mutex> lock(mMutex);
        while(mnNext < mnItems)
        {
            const int i = mnNext++;
            lock.unlock();
            std::exception_ptr pThrown;
            try
            {
                f(i);
            }
            catch(...)
            {
                pThrown = std::current_exception();
            }
            lock.lock();
            if(pThrown && !mpThrown)
                mpThrown = pThrown;
        }
        mDone.wait(lock, [this] { return mnRunning == 0; });
        mpJob = nullptr;
        if(mpThrown)
            std::rethrow_exception(mpThrown);
    }

    void WorkerPool::Work()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        while(true)
        {
            mWake.wait(lock, [this] { return mbStop || (mpJob && mnNext < mnItems); });
            if(mbStop)
                return;
            const std::function<void(int)>* const pJob = mpJob;
            const int i = mnNext++;
            ++mnRunning;
            lock.unlock();
            std::exception_ptr pThrown;
            try
            {
                (*pJob)(i);
            }
            catch(...)
            {
                pThrown = std::current_exception();
            }
            lock.lock();
            if(pThrown && !mpThrown)
                mpThrown = pThrown;
            if(--mnRunning == 0 && mnNext >= mnItems)
                mDone.notify_one();
        }
    }

} // namespace ORB_SLAM3
