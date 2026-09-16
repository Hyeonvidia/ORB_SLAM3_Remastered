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


#include "Viewer.hpp"
#include <pangolin/pangolin.h>
#include <pangolin/display/process.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>

#include <iostream>
#include <stdexcept>
#include <string>

namespace ORB_SLAM3
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const std::string &strSettingPath, Settings* settings):
    both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(std::exception &e)
            {

            }
        }
    }

    mbStopTrack = false;
}

void Viewer::newParameterLoader(Settings *settings) {
    mImageViewerScale = 1.f;

    float fps = settings->fps();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::Size imSize = settings->newImSize();
    mImageHeight = imSize.height;
    mImageWidth = imSize.width;

    mImageViewerScale = settings->imageViewerScale();
    mViewpointX = settings->viewPointX();
    mViewpointY = settings->viewPointY();
    mViewpointZ = settings->viewPointZ();
    mViewpointF = settings->viewPointF();
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    mImageViewerScale = 1.f;

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::FileNode node = fSettings["Camera.width"];
    if(!node.empty())
    {
        mImageWidth = node.real();
    }
    else
    {
        std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty())
    {
        mImageHeight = node.real();
    }
    else
    {
        std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.imageViewScale"];
    if(!node.empty())
    {
        mImageViewerScale = node.real();
    }

    node = fSettings["Viewer.ViewpointX"];
    if(!node.empty())
    {
        mViewpointX = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointY"];
    if(!node.empty())
    {
        mViewpointY = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointZ"];
    if(!node.empty())
    {
        mViewpointZ = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointF"];
    if(!node.empty())
    {
        mViewpointF = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    // One window now holds both the 3D map and the tracked frame;
    // 1600x900 leaves each a usable slot beside the menu panel.
    const int kWindowWidth = 1600, kWindowHeight = 900;
    pangolin::CreateWindowAndBind("ORB-SLAM3: Viewer",kWindowWidth,kWindowHeight);

    // Pangolin sizes its root view ONLY from an X11 ConfigureNotify
    // (thirdparty/Pangolin/components/pango_windowing/src/display_x11.cpp:422).
    // With no window manager -- Xvfb inside a container -- a window created
    // at its final size never receives one, so the root view stays 0x0, every
    // child view is empty and the window renders blank.
    //
    // Upstream only escaped this by accident: opening the separate
    // cv::imshow window restacked the display, and the resulting
    // ConfigureNotify is what sized Pangolin. Folding that window into this
    // one removed the accident, so the root view is now sized explicitly.
    pangolin::process::Resize(kWindowWidth, kWindowHeight);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Pangolin does not clip panel text -- Panel::Render disables the
    // scissor test outright (pango_display/src/widgets.cpp:269) -- so a
    // panel narrower than its longest label spills onto the 3D view.
    //
    // A checkbox label starts 28 px in (6 px panel inset + 18 px box +
    // 4 px gap) and the default font, AnonymousPro at 18 px, is
    // monospaced at 9.826 px per character. "Show Inertial Graph" is 19
    // characters, so it ends at 28 + 186.7 = 214.7 px -- 39.7 px past the
    // 175 px panel ORB-SLAM3 asked for. 221 px is the minimum that keeps
    // the same 6 px margin on the right; 240 leaves room for a 20th.
    const int kMenuPanelWidth = 240;
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(kMenuPanelWidth));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    pangolin::Var<bool> menuStop("menu.Stop",false,false);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step",false,true);  // false, true
    pangolin::Var<bool> menuStep("menu.Step",false,false);

    pangolin::Var<bool> menuShowOptLba("menu.Show LBA opt", false, true);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Both views are top-level displays with explicit bounds, the way
    // ORB-SLAM3 already placed its single one. A LayoutEqualHorizontal
    // container with AddDisplay() also reads well but renders nothing
    // here, so the arrangement stays explicit.
    //
    // The aspect sign matters: POSITIVE fits the view inside its bounds
    // (letterbox), NEGATIVE overfits and grows past them. ORB-SLAM3 used
    // a negative aspect, which was harmless when the 3D view owned the
    // whole window but covers its neighbours once it shares one.
    // (View::Resize, thirdparty/Pangolin/components/pango_display/src/view.cpp:75)
    // Where the 3D map ends and the tracked frame begins, as a fraction
    // of window width. Recomputed from the image once its size is known
    // (see below); this is only the value used before the first frame.
    // ORBSLAM3R_MAP_VIEW_FRACTION pins it instead, without a rebuild.
    double dMapViewFraction = 0.72;
    bool bMapFractionPinned = false;
    if(const char* f = std::getenv("ORBSLAM3R_MAP_VIEW_FRACTION"))
    {
        const double v = std::atof(f);
        if(v > 0.2 && v < 0.95) { dMapViewFraction = v; bMapFractionPinned = true; }
    }
    const double kMapViewRight = dMapViewFraction;

    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(kMenuPanelWidth),
                       kMapViewRight, 1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    // The tracked frame, which used to be a separate cv::imshow window.
    // Its aspect comes from the first frame, since it depends on the
    // sensor and on whether the right image is concatenated.
    pangolin::View& d_img = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, kMapViewRight, 1.0);

    pangolin::GlTexture imageTexture;
    int nLastImageCols = 0, nLastImageRows = 0;

    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();

    bool bFollow = true;
    bool bLocalizationMode = false;
    bool bStepByStep = false;
    bool bCameraView = true;

    if(mpTracker->mSensor == mpSystem->MONOCULAR || mpTracker->mSensor == mpSystem->STEREO || mpTracker->mSensor == mpSystem->RGBD)
    {
        menuShowGraph = true;
    }

    float trackedImageScale = mpTracker->GetImageScale();

    std::cout << "Starting the Viewer" << std::endl;
    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc,Ow);

        if(mbStopTrack)
        {
            menuStepByStep = true;
            mbStopTrack = false;
        }

        if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
        }

        if(menuTopView && mpMapDrawer->mpAtlas->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0));
            s_cam.Follow(Ow);
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        if(menuStepByStep && !bStepByStep)
        {
            //cout << "Viewer: step by step" << endl;
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        }
        else if(!menuStepByStep && bStepByStep)
        {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if(menuStep)
        {
            mpTracker->mbStep = true;
            menuStep = false;
        }


        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph || menuShowInertialGraph || menuShowOptLba)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph, menuShowInertialGraph, menuShowOptLba);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        // The frame is drawn before FinishFrame(), which swaps buffers.
        cv::Mat toShow;
        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

        if(both){
            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
            cv::hconcat(im,imRight,toShow);
        }
        else{
            toShow = im;
        }

        // This used to shrink the highgui window's contents. The frame
        // now scales with its view, so the setting only picks the
        // texture resolution.
        if(mImageViewerScale != 1.f)
        {
            int width = toShow.cols * mImageViewerScale;
            int height = toShow.rows * mImageViewerScale;
            cv::resize(toShow, toShow, cv::Size(width, height));
        }

        if(!toShow.empty())
        {
            if(toShow.cols != nLastImageCols || toShow.rows != nLastImageRows)
            {
                // The frame changes size when the right image is
                // concatenated, so the texture and the view's aspect
                // both follow it.
                imageTexture.Reinitialise(toShow.cols, toShow.rows, GL_RGB8,
                                          false, 0, GL_BGR, GL_UNSIGNED_BYTE);
                d_img.SetAspect(static_cast<double>(toShow.cols) / toShow.rows);

                // Size the frame view from the image's own resolution.
                // FrameDrawer burns its status line into the image with
                // FONT_HERSHEY_PLAIN at scale 1, roughly ten pixels tall,
                // so displaying below 1:1 resamples the text into mush.
                // Give the frame its native width where the window can
                // spare it and let the map have the rest.
                if(!bMapFractionPinned)
                {
                    const int nWinWidth = pangolin::DisplayBase().v.w > 0
                            ? pangolin::DisplayBase().v.w : kWindowWidth;
                    const double dFrameFrac = std::min(
                            0.55, static_cast<double>(toShow.cols) / nWinWidth);
                    const double dSplit = 1.0 - dFrameFrac;
                    d_cam.SetBounds(0.0, 1.0,
                                    pangolin::Attach::Pix(kMenuPanelWidth),
                                    dSplit, 1024.0f/768.0f);
                    d_img.SetBounds(0.0, 1.0, dSplit, 1.0);
                    d_img.SetAspect(static_cast<double>(toShow.cols) / toShow.rows);
                }

                nLastImageCols = toShow.cols;
                nLastImageRows = toShow.rows;
            }

            // Upload reads rows*cols*3 bytes contiguously, so a Mat that
            // is a view into a larger buffer has to be compacted first.
            const cv::Mat contiguous = toShow.isContinuous() ? toShow : toShow.clone();
            imageTexture.Upload(contiguous.data, GL_BGR, GL_UNSIGNED_BYTE);

            d_img.Activate();
            glColor3f(1.0f,1.0f,1.0f);
            // cv::Mat rows run top-down; OpenGL texture rows bottom-up.
            imageTexture.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();

        // cv::waitKey(mT) used to pace this loop as well as pump the
        // highgui event queue. FinishFrame() does not sleep, so the
        // pacing is explicit now.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(mT)));

        if(menuReset)
        {
            menuShowGraph = true;
            menuShowInertialGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->ResetActiveMap();
            menuReset = false;
        }

        if(menuStop)
        {
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();

            // Stop all threads
            mpSystem->Shutdown();

            // Save camera trajectory
            mpSystem->SaveTrajectoryEuRoC("CameraTrajectory.txt");
            mpSystem->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
            menuStop = false;
        }

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    std::unique_lock<std::mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopped = false;
}

/*void Viewer::SetTrackingPause()
{
    mbStopTrack = true;
}*/

}
