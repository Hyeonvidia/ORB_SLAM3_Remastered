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

#include "viewer/Viewer.hpp"
#include <pangolin/pangolin.h>
#include <pangolin/display/process.h>
#include <pangolin/display/default_font.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>
#include <streambuf>
#include <thread>

#include <iostream>
#include <stdexcept>
#include <string>
#include "viewer/MapDrawer.hpp"
#include "common/Settings.hpp"
#include "System.hpp"
#include "tracking/Tracking.hpp"

namespace ORB_SLAM3
{

    namespace
    {

        // Copies everything written to a stream into a bounded ring of recent lines,
        // while still forwarding it to wherever it was going. Installed on std::cout by
        // the viewer so it can show what the system is reporting; stdout, and so
        // run.log, is unchanged.
        class ViewerLogTap : public std::streambuf
        {
        public:
            explicit ViewerLogTap(std::ostream &stream, std::size_t nMaxLines = 300)
                : mStream(stream), mpOriginal(stream.rdbuf()), mnMaxLines(nMaxLines)
            {
                mStream.rdbuf(this);
            }

            ~ViewerLogTap() override { mStream.rdbuf(mpOriginal); }

            ViewerLogTap(const ViewerLogTap &) = delete;
            ViewerLogTap &operator=(const ViewerLogTap &) = delete;

            std::vector<std::string> Tail(std::size_t n) const
            {
                std::unique_lock<std::mutex> lock(mMutex);
                if(mvLines.size() <= n)
                    return mvLines;
                return std::vector<std::string>(mvLines.end() - n, mvLines.end());
            }

        protected:
            int overflow(int c) override
            {
                if(c == EOF)
                    return c;

                // Forward first, so a crash in the ring buffer cannot swallow output.
                mpOriginal->sputc(static_cast<char>(c));

                std::unique_lock<std::mutex> lock(mMutex);
                if(c == '\n')
                {
                    if(!msPartial.empty())
                    {
                        mvLines.push_back(msPartial);
                        msPartial.clear();
                        if(mvLines.size() > mnMaxLines)
                            mvLines.erase(mvLines.begin());
                    }
                }
                else if(c != '\r')
                {
                    msPartial.push_back(static_cast<char>(c));
                }
                return c;
            }

            int sync() override { return mpOriginal->pubsync(); }

        private:
            std::ostream &mStream;
            std::streambuf* mpOriginal;
            std::size_t mnMaxLines;
            mutable std::mutex mMutex;
            std::vector<std::string> mvLines;
            std::string msPartial;
        };

    } // namespace

    Viewer::Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking* pTracking,
                   const std::string &strSettingPath, Settings* settings)
        : both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpTracker(pTracking),
          mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
    {
        if(settings)
        {
            newParameterLoader(settings);
        }
        else
        {
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

    void Viewer::newParameterLoader(Settings* settings)
    {
        mImageViewerScale = 1.f;

        float fps = settings->fps();
        if(fps < 1)
            fps = 30;
        mT = 1e3 / fps;

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
        if(fps < 1)
            fps = 30;
        mT = 1e3 / fps;

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
        pangolin::CreateWindowAndBind("ORB-SLAM3: Viewer", kWindowWidth, kWindowHeight);

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
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kMenuPanelWidth));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuCamView("menu.Camera View", false, false);
        pangolin::Var<bool> menuTopView("menu.Top View", false, false);
        // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph", false, true);
        // Off by default: the inertial graph only has anything to draw on an
        // inertial sequence, and on the others it was a checked box doing nothing.
        pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial Graph", false, true);
        pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false, true);
        pangolin::Var<bool> menuReset("menu.Reset", false, false);
        pangolin::Var<bool> menuStop("menu.Stop", false, false);
        pangolin::Var<bool> menuStepByStep("menu.Step By Step", false, true); // false, true
        pangolin::Var<bool> menuStep("menu.Step", false, false);

        // On by default instead: this marks the keyframes local bundle adjustment
        // is currently touching, which is the part of the map actually moving.
        pangolin::Var<bool> menuShowOptLba("menu.Show LBA opt", true, true);
        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

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
        // THE LAYOUT: three rows to the right of the menu, stacked.
        //
        //   +------+---------------------------+
        //   |      |  tracked frame            |
        //   | menu +---------------------------+
        //   |      |  3D map                   |
        //   |      +---------------------------+
        //   |      |  log                      |
        //   +------+---------------------------+
        //
        // Side by side was worse and the reason is visible the moment a wide
        // sensor is used: a KITTI frame is 1226x370, so a column wide enough to
        // show it left two thirds of that column empty underneath, while the map
        // was squeezed into what remained on the left. Stacking gives the map the
        // full window width -- which is also the shape a driving trajectory wants
        // -- and leaves no dead region.
        //
        // The frame keeps its own aspect inside its row, so a tall frame (EuRoC
        // is 752x480) is centred with margins rather than stretched. Those
        // margins are the price of the arrangement; kMaxFrameFraction is what
        // stops a tall frame from taking the window, because the map is the view
        // being watched. ORBSLAM3R_FRAME_VIEW_FRACTION pins it without a rebuild.
        const double kMaxFrameFraction = 0.40;
        double dFrameFraction = 0.30;
        bool bFrameFractionPinned = false;
        if(const char* f = std::getenv("ORBSLAM3R_FRAME_VIEW_FRACTION"))
        {
            const double v = std::atof(f);
            if(v > 0.05 && v < 0.8)
            {
                dFrameFraction = v;
                bFrameFractionPinned = true;
            }
        }

        // Enough for a handful of recent lines; the map gets everything else.
        const int kLogHeightPx = 96;

        // The tracked frame's status line gets a row of its own rather than being
        // drawn into the image. Burned in, it is part of the texture and shrinks
        // with the frame -- which is what made it illegible once the frame was
        // scaled to fit its row -- and the band it needed ate image rows, which
        // then fed back into the layout. Drawn here it is at window resolution,
        // always, and the image stays an image.
        const int kStatusHeightPx = 26;

        // The tracked frame, which used to be a separate cv::imshow window.
        // Its aspect comes from the first frame, since it depends on the sensor
        // and on whether the right image is concatenated.
        pangolin::View &d_img = pangolin::CreateDisplay().SetBounds(1.0 - dFrameFraction, 1.0,
                                                                    pangolin::Attach::Pix(kMenuPanelWidth), 1.0);

        // No aspect argument, deliberately: a positive aspect letterboxes, which
        // would hand back most of the width this layout exists to give the map.
        // The projection is what follows the view's shape instead; see
        // MapProjection below.
        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(pangolin::Attach::Pix(kLogHeightPx),
                                               1.0 - dFrameFraction -
                                                   static_cast<double>(kStatusHeightPx) / kWindowHeight,
                                               pangolin::Attach::Pix(kMenuPanelWidth), 1.0)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

        // ProjectionMatrix's first two arguments are the viewport it is built
        // for. They were fixed at 1024x768 no matter what the layout gave the
        // view, which stretched the map horizontally the moment the view stopped
        // being 4:3. Remembering the focal length and far plane in use lets the
        // projection be rebuilt whenever the view is resized.
        double dProjFocal = mViewpointF, dProjFar = 1000.0;
        int nLastCamW = 0, nLastCamH = 0;
        auto MapProjection = [&d_cam, &dProjFocal, &dProjFar](double focal, double far)
        {
            dProjFocal = focal;
            dProjFar = far;
            const int w = d_cam.v.w > 0 ? d_cam.v.w : 1024;
            const int h = d_cam.v.h > 0 ? d_cam.v.h : 768;
            return pangolin::ProjectionMatrix(w, h, focal, focal, w / 2.0, h / 2.0, 0.1, far);
        };

        // Given real bounds here, not left at zero height like the log: the block
        // below that resizes the rows only runs when the frame fraction is being
        // derived, so a pinned ORBSLAM3R_FRAME_VIEW_FRACTION would otherwise
        // leave this row invisible.
        const double kInitStatusFrac = static_cast<double>(kStatusHeightPx) / kWindowHeight;
        pangolin::View &d_status = pangolin::CreateDisplay().SetBounds(
            1.0 - dFrameFraction - kInitStatusFrac, 1.0 - dFrameFraction, pangolin::Attach::Pix(kMenuPanelWidth), 1.0);

        pangolin::View &d_log = pangolin::CreateDisplay().SetBounds(0.0, pangolin::Attach::Pix(kLogHeightPx),
                                                                    pangolin::Attach::Pix(kMenuPanelWidth), 1.0);

        // Installed for the lifetime of the viewer, then std::cout is
        // restored.
        ViewerLogTap logTap(std::cout);

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

        if(mpTracker->mSensor == mpSystem->MONOCULAR || mpTracker->mSensor == mpSystem->STEREO ||
           mpTracker->mSensor == mpSystem->RGBD)
        {
            menuShowGraph = true;
        }

        float trackedImageScale = mpTracker->GetImageScale();

        std::cout << "Starting the Viewer" << std::endl;
        while(1)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc, Ow);

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
                    s_cam.SetProjectionMatrix(MapProjection(mViewpointF, 1000));
                    s_cam.SetModelViewMatrix(
                        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
                    s_cam.Follow(Twc);
                }
                else
                {
                    s_cam.SetProjectionMatrix(MapProjection(3000, 1000));
                    s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0.01, 10, 0, 0, 0, 0.0, 0.0, 1.0));
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
                s_cam.SetProjectionMatrix(MapProjection(mViewpointF, 10000));
                s_cam.SetModelViewMatrix(
                    pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
            }

            if(menuTopView && mpMapDrawer->mpAtlas->isImuInitialized())
            {
                menuTopView = false;
                bCameraView = false;
                s_cam.SetProjectionMatrix(MapProjection(3000, 10000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
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

            // The row changes height when the frame's aspect is first known, and
            // again if the window is resized; the projection has to follow.
            if(d_cam.v.w != nLastCamW || d_cam.v.h != nLastCamH)
            {
                nLastCamW = d_cam.v.w;
                nLastCamH = d_cam.v.h;
                s_cam.SetProjectionMatrix(MapProjection(dProjFocal, dProjFar));
            }

            d_cam.Activate(s_cam);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            mpMapDrawer->DrawCurrentCamera(Twc);
            if(menuShowKeyFrames || menuShowGraph || menuShowInertialGraph || menuShowOptLba)
                mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph, menuShowInertialGraph, menuShowOptLba);
            if(menuShowPoints)
                mpMapDrawer->DrawMapPoints();

            // The frame is drawn before FinishFrame(), which swaps buffers.
            cv::Mat toShow;
            cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

            if(both)
            {
                cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
                cv::hconcat(im, imRight, toShow);
            }
            else
            {
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
                    imageTexture.Reinitialise(toShow.cols, toShow.rows, GL_RGB8, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
                    d_img.SetAspect(static_cast<double>(toShow.cols) / toShow.rows);

                    // Height of the top row, from the image's own resolution.
                    // FrameDrawer burns its status line into the image with
                    // FONT_HERSHEY_PLAIN at scale 1, roughly ten pixels tall, so
                    // displaying below 1:1 resamples the text into mush: ask for
                    // the height the frame needs at the full window width, and
                    // take the cap only when that would crowd out the map.
                    if(!bFrameFractionPinned)
                    {
                        const int nWinWidth = pangolin::DisplayBase().v.w > 0 ? pangolin::DisplayBase().v.w
                                                                              : kWindowWidth;
                        const int nWinHeight = pangolin::DisplayBase().v.h > 0 ? pangolin::DisplayBase().v.h
                                                                               : kWindowHeight;
                        const int nRowWidth = std::max(1, nWinWidth - kMenuPanelWidth);
                        const double dNatural = static_cast<double>(toShow.rows) * nRowWidth / toShow.cols;
                        dFrameFraction = std::min(kMaxFrameFraction, dNatural / nWinHeight);

                        const double dStatusFrac = static_cast<double>(kStatusHeightPx) / nWinHeight;
                        d_img.SetBounds(1.0 - dFrameFraction, 1.0, pangolin::Attach::Pix(kMenuPanelWidth), 1.0);
                        d_status.SetBounds(1.0 - dFrameFraction - dStatusFrac, 1.0 - dFrameFraction,
                                           pangolin::Attach::Pix(kMenuPanelWidth), 1.0);
                        d_cam.SetBounds(pangolin::Attach::Pix(kLogHeightPx), 1.0 - dFrameFraction - dStatusFrac,
                                        pangolin::Attach::Pix(kMenuPanelWidth), 1.0);
                    }
                    d_img.SetAspect(static_cast<double>(toShow.cols) / toShow.rows);

                    nLastImageCols = toShow.cols;
                    nLastImageRows = toShow.rows;
                }

                // Upload reads rows*cols*3 bytes contiguously, so a Mat that
                // is a view into a larger buffer has to be compacted first.
                const cv::Mat contiguous = toShow.isContinuous() ? toShow : toShow.clone();

                // GL_UNPACK_ALIGNMENT defaults to 4: OpenGL expects every row of
                // the source to start on a 4-byte boundary. A cv::Mat packs its
                // rows with no padding at all, so whenever cols*3 is not a
                // multiple of 4 the reader slips a little further into the next
                // row each time and the picture shears diagonally, with the last
                // rows running off the end of the buffer. KITTI is 1226 wide --
                // 3678 bytes a row, 2 over -- which is why it showed the tilt
                // while EuRoC (752) and TUM (640) did not.
                GLint nUnpackAlign = 4;
                glGetIntegerv(GL_UNPACK_ALIGNMENT, &nUnpackAlign);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                imageTexture.Upload(contiguous.data, GL_BGR, GL_UNSIGNED_BYTE);
                glPixelStorei(GL_UNPACK_ALIGNMENT, nUnpackAlign);

                d_img.Activate();
                glColor3f(1.0f, 1.0f, 1.0f);
                // cv::Mat rows run top-down; OpenGL texture rows bottom-up.
                imageTexture.RenderToViewportFlipY();
            }

            // The status row: what the system is doing on the left, how the map is
            // being looked at on the right. The view mode belongs here rather than
            // floating over the map, because "Camera View" and "Top View" are
            // momentary buttons -- they pop back up and leave nothing on screen
            // saying which one is in effect.
            if(d_status.v.h > 0)
            {
                d_status.Activate();
                pangolin::GlFont &statusFont = pangolin::default_font();
                const float fY = static_cast<float>(d_status.v.b) + (d_status.v.h - statusFont.Height()) * 0.5f + 1.0f;

                glColor3f(0.10f, 0.10f, 0.10f);
                statusFont.Text(mpFrameDrawer->StatusText()).DrawWindow(static_cast<float>(d_status.v.l) + 8.0f, fY);

                const std::string sMode = std::string(bCameraView ? "Camera View" : "Top View") +
                                          (menuFollowCamera ? "  |  following" : "  |  free look");
                pangolin::GlText mode = statusFont.Text(sMode);
                glColor3f(0.45f, 0.45f, 0.45f);
                mode.DrawWindow(static_cast<float>(d_status.v.l + d_status.v.w) - mode.Width() - 10.0f, fY);
            }

            // The log, newest line at the bottom, in the space under the frame.
            if(d_log.v.h > 0)
            {
                d_log.Activate();
                pangolin::GlFont &font = pangolin::default_font();
                const float fLineHeight = font.Height() + 2.0f;
                const int nLines = std::max(1, static_cast<int>(d_log.v.h / fLineHeight) - 1);
                const std::vector<std::string> vLines = logTap.Tail(nLines);
                glColor3f(0.15f, 0.15f, 0.15f);
                float fY = static_cast<float>(d_log.v.b) + fLineHeight * (vLines.size() - 1) + 4.0f;
                for(const std::string &line : vLines)
                {
                    font.Text(line).DrawWindow(static_cast<float>(d_log.v.l) + 6.0f, fY);
                    fY -= fLineHeight;
                }
            }

            pangolin::FinishFrame();

            // cv::waitKey(mT) used to pace this loop as well as pump the
            // highgui event queue. FinishFrame() does not sleep, so the
            // pacing is explicit now.
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(mT)));

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

} // namespace ORB_SLAM3
