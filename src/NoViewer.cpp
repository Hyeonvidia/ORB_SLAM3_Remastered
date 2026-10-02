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

// The viewer of a library built without one (ORBSLAM3R_BUILD_VIEWER=OFF):
// drawers that take what Tracking tells them and do nothing with it, and no
// viewer thread. Pangolin and OpenGL are then neither compiled against nor
// linked.

#include "ViewerPort.hpp"

#include "tracking/ViewPorts.hpp"

#include <iostream>
#include <string>

namespace ORB_SLAM3
{

    namespace
    {
        class NoFrameView : public FrameViewPort
        {
        public:
            void Update(Tracking*) override {}
            void SetBoth(bool) override {}
        };

        class NoMapView : public MapViewPort
        {
        public:
            void SetCurrentCameraPose(const Sophus::SE3f &) override {}
        };
    } // namespace

    FrameViewPort* MakeFrameView(Atlas*)
    {
        return new NoFrameView;
    }

    MapViewPort* MakeMapView(Atlas*, const std::string &, Settings*)
    {
        return new NoMapView;
    }

    ViewerPort* MakeViewer(System*, FrameViewPort*, MapViewPort*, Tracking*, const std::string &, Settings*)
    {
        std::cout << "Viewer requested, but this library was built without one (ORBSLAM3R_BUILD_VIEWER=OFF)."
                  << std::endl;
        return nullptr;
    }

} // namespace ORB_SLAM3
