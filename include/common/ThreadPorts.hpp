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

#ifndef THREADPORTS_H
#define THREADPORTS_H

#include <atomic>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    namespace IMU
    {
        class Bias;
    }

    // The three threads of the paper -- Tracking, Local Mapping, Loop and Map
    // Merging -- hand keyframes down the line and reach back up it to stop,
    // reset and release each other. In v1.0 each held a pointer to the
    // others' classes, so no one of them could be read, built or tested
    // without the other two. What each needs of another is small; these are
    // those needs, and nothing more. A thread class implements its port and
    // holds the ports of the others.

    // The states Tracking reports. Here rather than in Tracking because Local
    // Mapping asks about them.
    struct TrackingStates
    {
        enum eTrackingState
        {
            SYSTEM_NOT_READY = -1,
            NO_IMAGES_YET = 0,
            NOT_INITIALIZED = 1,
            OK = 2,
            RECENTLY_LOST = 3,
            LOST = 4,
            OK_KLT = 5
        };
    };

    // A pointer that one thread writes and another reads: atomic, and read
    // through -> as a plain pointer is.
    template<class T>
    class SharedPointer
    {
    public:
        SharedPointer(T* p = nullptr) : mp(p) {}
        SharedPointer &operator=(T* p)
        {
            mp.store(p);
            return *this;
        }
        operator T*() const { return mp.load(); }
        T* operator->() const { return mp.load(); }

    private:
        std::atomic<T*> mp;
    };

    // What Local Mapping and Loop Closing ask of Tracking.
    class TrackerPort : public TrackingStates
    {
    public:
        virtual ~TrackerPort() = default;

        virtual eTrackingState State() const = 0;
        virtual void SetState(eTrackingState state) = 0;
        virtual KeyFrame* GetLastKeyFrame() = 0;
        virtual int GetMatchesInliers() = 0;
        // The timestamps of the frame being tracked and of the one before, as
        // they were when tracking the current one began.
        virtual double CurrentFrameTime() const = 0;
        virtual double LastFrameTime() const = 0;
        // After an IMU initialisation or a loop: the scale and the bias the
        // frames since the keyframe are to take.
        virtual void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame) = 0;
    };

    // What Tracking and Loop Closing ask of Local Mapping.
    class MapperPort
    {
    public:
        virtual ~MapperPort() = default;

        virtual void InsertKeyFrame(KeyFrame* pKF) = 0;
        virtual bool AcceptKeyFrames() = 0;
        virtual bool SetNotStop(bool flag) = 0;
        virtual void InterruptBA() = 0;
        virtual int KeyframesInQueue() = 0;
        virtual bool IsInitializing() = 0;
        virtual bool BadImu() = 0;
        virtual void SetFirstTimestamp(double ts) = 0;
        virtual void RequestReset() = 0;
        virtual void RequestResetActiveMap(Map* pMap) = 0;
        virtual void RequestStop() = 0;
        virtual bool stopRequested() = 0;
        virtual bool isStopped() = 0;
        virtual void Release() = 0;
        virtual void EmptyQueue() = 0;
        virtual bool isFinished() = 0;
    };

    // What Tracking and Local Mapping ask of Loop Closing.
    class LoopCloserPort
    {
    public:
        virtual ~LoopCloserPort() = default;

        virtual void InsertKeyFrame(KeyFrame* pKF) = 0;
        virtual void RequestReset() = 0;
        virtual void RequestResetActiveMap(Map* pMap) = 0;
    };

} // namespace ORB_SLAM3

#endif // THREADPORTS_H
