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

#ifndef VIEWPORTS_H
#define VIEWPORTS_H

#include <sophus/se3.hpp>

namespace ORB_SLAM3
{

    class Tracking;

    // What Tracking tells whoever draws. In v1.0 it held the viewer's two
    // drawers themselves, so the tracking layer included the viewer's headers
    // and, through them, Pangolin's; a build without a viewer was not possible.
    // The drawers implement these, and a library built without the viewer has
    // ones that do nothing.

    // The image of the frame just tracked and what was found in it.
    class FrameViewPort
    {
    public:
        virtual ~FrameViewPort() = default;

        // The tracker's last frame is copied; called once a frame.
        virtual void Update(Tracking* pTracker) = 0;
        // Two cameras that are not rectified: both images are shown.
        virtual void SetBoth(bool bBoth) = 0;
    };

    // The pose the map is drawn from.
    class MapViewPort
    {
    public:
        virtual ~MapViewPort() = default;

        virtual void SetCurrentCameraPose(const Sophus::SE3f &Tcw) = 0;
    };

} // namespace ORB_SLAM3

#endif // VIEWPORTS_H
