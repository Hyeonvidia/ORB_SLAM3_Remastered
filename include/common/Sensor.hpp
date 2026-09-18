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

#ifndef SENSOR_H
#define SENSOR_H

namespace ORB_SLAM3
{

    // The sensor configuration a system runs with.
    //
    // It used to be System::eSensor, nested in the class at the top of the
    // stack -- so a file that only needed to ask "is this stereo?" had to
    // include System.hpp, which includes tracking, local mapping, loop closing
    // and the atlas. common/Settings.cpp, at the bottom of the stack, was one of
    // them. System keeps the old spelling as aliases, so System::MONOCULAR in
    // the examples and elsewhere is unchanged.
    struct Sensor
    {
        enum eSensor
        {
            MONOCULAR = 0,
            STEREO = 1,
            RGBD = 2,
            IMU_MONOCULAR = 3,
            IMU_STEREO = 4,
            IMU_RGBD = 5,
        };
    };

} // namespace ORB_SLAM3

#endif // SENSOR_H
