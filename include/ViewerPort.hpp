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

#ifndef VIEWERPORT_H
#define VIEWERPORT_H

#include <string>

namespace ORB_SLAM3
{

    class Atlas;
    class FrameViewPort;
    class MapViewPort;
    class Settings;
    class System;
    class Tracking;

    // What System asks of the viewer it runs on a thread of its own.
    class ViewerPort
    {
    public:
        virtual ~ViewerPort() = default;

        // The thread's function: draws until asked to finish.
        virtual void Run() = 0;
        virtual void RequestFinish() = 0;
        virtual void RequestStop() = 0;
        virtual bool isFinished() = 0;
        virtual bool isStopped() = 0;
        virtual void Release() = 0;
    };

    // The viewer, as the library was built: with ORBSLAM3R_BUILD_VIEWER these
    // are src/viewer/ViewerFactory.cpp and make the Pangolin viewer; without,
    // src/NoViewer.cpp, whose drawers do nothing and whose viewer is null.
    // System includes this header and none of the viewer's, so the viewer is
    // above System and can be left out of the build.
    FrameViewPort* MakeFrameView(Atlas* pAtlas);
    MapViewPort* MakeMapView(Atlas* pAtlas, const std::string &strSettingPath, Settings* settings);
    // The two views are the ones the functions above made.
    ViewerPort* MakeViewer(System* pSystem, FrameViewPort* pFrameView, MapViewPort* pMapView, Tracking* pTracking,
                           const std::string &strSettingPath, Settings* settings);

} // namespace ORB_SLAM3

#endif // VIEWERPORT_H
