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

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "atlas/MapPoint.hpp"
#include "atlas/Atlas.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>

#include <map>
#include <utility>
#include <vector>

namespace ORB_SLAM3
{

    class Tracking;

    class FrameDrawer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        FrameDrawer(Atlas* pAtlas);

        // Update info from the last processed frame.
        void Update(Tracking* pTracker);

        // Draw last processed frame.
        cv::Mat DrawFrame(float imageScale = 1.f);
        cv::Mat DrawRightFrame(float imageScale = 1.f);

        bool both;

        // What the system is doing, rebuilt from the Atlas on every frame. The
        // viewer draws it in a row of its own at window resolution; it used to be
        // written into the image with cv::putText, where it was part of the
        // texture and came apart whenever the frame was scaled under 1:1.
        std::string StatusText() const { return msStatusText; }

        // How many frames Tracking has handed over, and the state of the last
        // one -- what the viewer's log reports progress from. Atomics, so that
        // asking costs the tracking thread nothing.
        int FrameCount() const { return mnFrames.load(std::memory_order_relaxed); }
        int TrackingState() const { return mnLastState.load(std::memory_order_relaxed); }

    protected:
        void UpdateStatusText(int nState);

        std::string msStatusText;

        std::atomic<int> mnFrames{0};
        std::atomic<int> mnLastState{-1}; // Tracking::SYSTEM_NOT_READY

        // System::eSensor, taken from Tracking. -1 until the first update.
        int mnSensor = -1;

        // Info of the frame to be drawn
        cv::Mat mIm, mImRight;
        int N;
        std::vector<cv::KeyPoint> mvCurrentKeys, mvCurrentKeysRight;
        std::vector<bool> mvbMap, mvbVO;
        bool mbOnlyTracking;
        int mnTracked, mnTrackedVO;
        std::vector<cv::KeyPoint> mvIniKeys;
        std::vector<int> mvIniMatches;
        int mState;
        std::vector<float> mvCurrentDepth;
        float mThDepth;

        Atlas* mpAtlas;

        std::mutex mMutex;
        std::vector<std::pair<cv::Point2f, cv::Point2f>> mvTracks;

        Frame mCurrentFrame;
        std::vector<MapPoint*> mvpLocalMap;
        std::vector<cv::KeyPoint> mvMatchedKeys;
        std::vector<MapPoint*> mvpMatchedMPs;
        std::vector<cv::KeyPoint> mvOutlierKeys;
        std::vector<MapPoint*> mvpOutlierMPs;

        std::map<long unsigned int, cv::Point2f> mmProjectPoints;
        std::map<long unsigned int, cv::Point2f> mmMatchedInImage;
    };

} // namespace ORB_SLAM3

#endif // FRAMEDRAWER_H
