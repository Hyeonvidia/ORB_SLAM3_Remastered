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

#ifndef ORBDESCRIPTOR_H
#define ORBDESCRIPTOR_H

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

    // The part of ORB matching that MapPoint and Frame use: comparing two
    // descriptors. It lives apart from ORBmatcher, which works on Frames,
    // KeyFrames and MapPoints, so that those classes can use it without
    // depending on the matcher.
    class ORBdescriptor
    {
    public:
        // Computes the Hamming distance between two ORB descriptors
        static int Distance(const cv::Mat &a, const cv::Mat &b);

        static constexpr int TH_LOW = 50;
        static constexpr int TH_HIGH = 100;
    };

} // namespace ORB_SLAM3

#endif // ORBDESCRIPTOR_H
