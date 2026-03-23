/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
* Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
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

#pragma once

#include "Atlas.hpp"
#include "KeyFrame.hpp"
#include "Tracking.hpp"
#include <string>

namespace ORB_SLAM3 {

class LocalMapping;
class System;

class TrajectoryWriter {
public:
    // sensor enum mirrored from System — avoids including System.hpp
    enum eSensor {
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };

    TrajectoryWriter(Atlas* pAtlas, Tracking* pTracker,
                     LocalMapping* pLocalMapper, eSensor sensor)
        : mpAtlas(pAtlas), mpTracker(pTracker),
          mpLocalMapper(pLocalMapper), mSensor(sensor) {}

    void SaveTrajectoryTUM(const std::string& filename);
    void SaveKeyFrameTrajectoryTUM(const std::string& filename);
    void SaveTrajectoryEuRoC(const std::string& filename);
    void SaveTrajectoryEuRoC(const std::string& filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const std::string& filename);
    void SaveKeyFrameTrajectoryEuRoC(const std::string& filename, Map* pMap);
    void SaveTrajectoryKITTI(const std::string& filename);
    void SaveDebugData(const int& initIdx);

private:
    Atlas* mpAtlas;
    Tracking* mpTracker;
    LocalMapping* mpLocalMapper;
    eSensor mSensor;
};

} // namespace ORB_SLAM3
