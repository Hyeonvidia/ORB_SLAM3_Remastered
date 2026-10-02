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

#include "ViewerPort.hpp"

#include "viewer/FrameDrawer.hpp"
#include "viewer/MapDrawer.hpp"
#include "viewer/Viewer.hpp"

#include <string>

namespace ORB_SLAM3
{

    FrameViewPort* MakeFrameView(Atlas* pAtlas)
    {
        return new FrameDrawer(pAtlas);
    }

    MapViewPort* MakeMapView(Atlas* pAtlas, const std::string &strSettingPath, Settings* settings)
    {
        return new MapDrawer(pAtlas, strSettingPath, settings);
    }

    ViewerPort* MakeViewer(System* pSystem, FrameViewPort* pFrameView, MapViewPort* pMapView, Tracking* pTracking,
                           const std::string &strSettingPath, Settings* settings)
    {
        // The views are the drawers the two functions above made.
        FrameDrawer* pFrameDrawer = static_cast<FrameDrawer*>(pFrameView);
        MapDrawer* pMapDrawer = static_cast<MapDrawer*>(pMapView);
        Viewer* pViewer = new Viewer(pSystem, pFrameDrawer, pMapDrawer, pTracking, strSettingPath, settings);
        pViewer->both = pFrameDrawer->both;
        return pViewer;
    }

} // namespace ORB_SLAM3
