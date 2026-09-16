#!/usr/bin/env python3
"""
Stage three of the port: targeted source fixes that neither the include rewrite
(port_body.py) nor the std:: qualification (qualify_std.py) can express.

Each fix states what it changes and why.  Keeping them here rather than editing
src/ by hand means the whole port replays from the pristine reference checkout:

    rm -rf src include && cp -R reference/ORB_SLAM3/{src,include} .
    ./tools/port_body.py && ./tools/qualify_std.py && ./tools/port_fixes.py

  ./tools/port_fixes.py --check
  ./tools/port_fixes.py
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
def fix_g2o_solver_ownership(text, path):
    """Route g2o solver construction through vendor_ext's factory.

    ORB-SLAM3 builds optimizers the way g2o worked in 2016, with raw owning
    pointers threaded through three constructors:

        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
        linearSolver = new g2o::LinearSolverEigen<...::PoseMatrixType>();
        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    Current g2o takes std::unique_ptr at both levels, so every one of these is a
    compile error -- 33 of them in this file.  MakeLevenberg/MakeGaussNewton
    collapse the whole preamble into one call and keep the ownership convention
    in a single header.

    A side effect worth noting: where the original creates one solver_ptr and
    then branches to build two algorithms from it (BlockSolverX in the bLarge
    case), only one algorithm is ever used and the other silently leaks the
    block solver.  Building the algorithm inside each branch removes that.
    """
    if path.name != "Optimizer.cpp":
        return text, 0

    lines = text.splitlines(keepends=True)
    out = []
    block_solver = None      # e.g. "g2o::BlockSolver_6_3"
    kind = "kEigen"
    n = 0

    decl_re = re.compile(
        r"^\s*(g2o::BlockSolver\w*)::LinearSolverType\s*\*\s*linearSolver\s*;")
    decl_assign_re = re.compile(
        r"^\s*(g2o::BlockSolver\w*)::LinearSolverType\s*\*\s*linearSolver\s*=")
    assign_re = re.compile(
        r"^\s*linearSolver\s*=\s*new\s+g2o::LinearSolver(Eigen|Dense)\s*<")
    inline_new_re = re.compile(
        r"^\s*new\s+g2o::LinearSolver(Eigen|Dense)\s*<")
    ptr_re = re.compile(
        r"^\s*(g2o::BlockSolver\w*)\s*\*\s*\w+\s*=\s*new\s+g2o::BlockSolver\w*\s*\(\s*linearSolver\s*\)\s*;")
    algo_re = re.compile(
        r"(g2o::OptimizationAlgorithm(Levenberg|GaussNewton))\s*\*\s*(\w+)\s*="
        r"\s*new\s+g2o::OptimizationAlgorithm(?:Levenberg|GaussNewton)\s*\(\s*\w+\s*\)\s*;")

    for line in lines:
        m = decl_re.match(line) or decl_assign_re.match(line)
        if m:
            block_solver = m.group(1)
            # A declaration that also assigns carries its kind on this line or
            # the next; assign_re/inline_new_re pick it up either way.
            km = re.search(r"LinearSolver(Eigen|Dense)", line)
            if km:
                kind = "kEigen" if km.group(1) == "Eigen" else "kDense"
            n += 1
            continue
        m = assign_re.match(line) or inline_new_re.match(line)
        if m:
            kind = "kEigen" if m.group(1) == "Eigen" else "kDense"
            n += 1
            continue
        m = ptr_re.match(line)
        if m:
            block_solver = m.group(1)
            n += 1
            continue
        m = algo_re.search(line)
        if m and block_solver:
            algo = "MakeLevenberg" if m.group(2) == "Levenberg" else "MakeGaussNewton"
            indent = re.match(r"^\s*", line).group(0)
            replacement = (
                f"{indent}auto* {m.group(3)} = orbslam3r::g2o_ext::{algo}<"
                f"{block_solver}, orbslam3r::g2o_ext::LinearSolver::{kind}>();\n")
            out.append(replacement)
            n += 1
            continue
        out.append(line)

    text = "".join(out)

    # Collapse the blank lines the removed declarations left behind.
    text = re.sub(r"\n{3,}", "\n\n", text)

    if n and "g2o_ext/solver_factory.hpp" not in text:
        text = text.replace(
            "#include <orbslam3r/g2o_ext/compat.hpp>",
            "#include <orbslam3r/g2o_ext/compat.hpp>\n"
            "#include <orbslam3r/g2o_ext/solver_factory.hpp>", 1)
    return text, n


# ---------------------------------------------------------------------------
def fix_full_ba_generation_counter(text, path):
    """`mnFullBAIdx` is incremented and compared, but declared bool.

    Three call sites abort a running global bundle adjustment with

        mbStopGBA = true;
        mnFullBAIdx++;

    and RunGlobalBundleAdjustment guards its map update with

        int idx = mnFullBAIdx;
        { lock(mMutexGBA); if (idx != mnFullBAIdx) return; ... }

    With `bool mnFullBAIdx;` the first `++` sets it to true and every later one
    is a no-op, so after the first abort in a session `idx != mnFullBAIdx` can
    never be true again and that guard is permanently dead. ORB-SLAM2 declared
    the same member `int`.

    C++17 removed operator++ on bool, which is the only reason this surfaced.

    SCOPE: only the type is changed here. ORB-SLAM3 also reads the snapshot
    *after* the optimisation returns, where ORB-SLAM2 read it before, so even as
    an int this guard now only covers the gap between that read and acquiring
    mMutexGBA. Moving the snapshot back would change runtime behaviour rather
    than fix a compile error, and the primary abort handling is mbStopGBA, which
    is checked separately a few lines below.
    """
    if path.stem != "LoopClosing" or path.suffix not in (".h", ".hpp"):
        return text, 0
    new, n = re.subn(
        r"^(\s*)bool(\s+mnFullBAIdx\s*;)",
        r"\1// Counter, not a flag: incremented on every GBA abort and compared\n"
        r"\1// with != in RunGlobalBundleAdjustment. As a bool it saturates at\n"
        r"\1// true and that comparison stops working. See tools/port_fixes.py.\n"
        r"\1int\2",
        text, flags=re.MULTILINE)
    return new, n


# ---------------------------------------------------------------------------
def add_dbow2_serialization(text, path):
    """KeyFrame archives DBoW2::BowVector and FeatureVector.

    ORB-SLAM3 made that work by adding an intrusive serialize() member to its
    forked copies of both classes.  vendor_ext supplies the same capability
    non-intrusively, but the overloads have to be visible where the archive is
    instantiated.
    """
    if path.stem != "KeyFrame" or path.suffix not in (".h", ".hpp"):
        return text, 0
    if "dbow2_ext/serialization.hpp" in text:
        return text, 0
    anchor = "#include <boost/serialization/map.hpp>"
    if anchor not in text:
        return text, 0
    return text.replace(
        anchor,
        anchor + "\n\n"
        "// Non-intrusive Boost support for DBoW2::BowVector / FeatureVector,\n"
        "// replacing the intrusive members ORB-SLAM3 added to its DBoW2 fork.\n"
        "#include <orbslam3r/dbow2_ext/serialization.hpp>", 1), 1


# ---------------------------------------------------------------------------
COMPILEDWITHC11_RE = re.compile(
    r"^[ \t]*#[ \t]*ifdef[ \t]+COMPILEDWITHC11[ \t]*\r?\n"
    r"(?P<cxx11>.*?)"
    r"^[ \t]*#[ \t]*else[ \t]*\r?\n"
    r".*?"
    r"^[ \t]*#[ \t]*endif[ \t]*\r?\n",
    re.MULTILINE | re.DOTALL,
)


def drop_compiledwithc11_guards(text, path):
    """Collapse `#ifdef COMPILEDWITHC11` to its C++11 branch.

    Upstream guards every timing call like this:

        #ifdef COMPILEDWITHC11
            auto t1 = std::chrono::steady_clock::now();
        #else
            auto t1 = std::chrono::monotonic_clock::now();
        #endif

    and relies on its CMake defining COMPILEDWITHC11.  The #else branch names
    std::chrono::monotonic_clock, which is a pre-standard name that never
    existed in C++11 or later -- so that branch cannot compile on any current
    toolchain, and the macro only ever hid it.  This project is C++17, so the
    guard is deleted and the live branch kept.
    """
    if not COMPILEDWITHC11_RE.search(text):
        return text, 0
    return COMPILEDWITHC11_RE.subn(lambda m: m.group("cxx11"), text)


# ---------------------------------------------------------------------------
def make_viewer_switchable(text, path):
    """Let ORBSLAM3R_VIEWER override the Pangolin viewer in both directions.

    Upstream's examples disagree about the flag they pass: mono_euroc asks for
    no viewer, stereo_euroc asks for one, and stereo_inertial_euroc asks for
    none again.  Inside a container the viewer runs on llvmpipe over Xvfb, where
    it drags a batch run to a crawl -- stereo_euroc on EuRoC MH01 made no
    measurable progress in twelve minutes with it on, versus three and a half
    for the whole sequence with it off.

    Upstream clearly hit this too; the line below the check is their own
    commented-out `if(false) // TODO`.

    The override has to work both ways. Forcing off makes batch runs usable;
    forcing on is the only way to exercise the viewer from an example that
    hard-codes bUseViewer=false, which is most of them. With the variable unset
    the caller's flag wins and behaviour is unchanged.
    """
    if path.name != "System.cpp":
        return text, 0
    old = "    //Initialize the Viewer thread and launch\n    if(bUseViewer)\n"
    if old not in text:
        return text, 0
    new = (
        "    //Initialize the Viewer thread and launch\n"
        "    // ORBSLAM3R_VIEWER overrides the caller's flag in both directions:\n"
        "    // \"0\" forces the viewer off, anything else forces it on, and unset\n"
        "    // leaves the decision to the caller. Both directions are needed\n"
        "    // because upstream's examples hard-code opposite flags, so without\n"
        "    // an override there is no way to run a given example the other way.\n"
        "    const char* viewerEnv = std::getenv(\"ORBSLAM3R_VIEWER\");\n"
        "    const bool bViewerEnabled =\n"
        "        viewerEnv ? (std::string(viewerEnv) != \"0\") : bUseViewer;\n"
        "    if(bViewerEnabled)\n"
    )
    text = text.replace(old, new, 1)
    if "#include <cstdlib>" not in text:
        text = text.replace("#include <iostream>",
                            "#include <cstdlib>\n#include <iostream>", 1)
    return text, 1


# ---------------------------------------------------------------------------
def fix_viewer_layout(text, path):
    """One window for both views, a panel wide enough for its labels, and
    Follow Camera on by default.

    Three problems with the stock viewer:

    1. It opens TWO OS windows -- a Pangolin window for the 3D map and a
       separate OpenCV highgui window for the tracked frame. Under a bare X
       server both land at +0+0 on top of each other, and neither respects the
       other's aspect ratio.
    2. The menu panel is 175 px wide, narrower than its longest labels.
       Pangolin does not clip panel text, so "Show Inertial Graph" and
       "Localization Mode" render straight past the grey panel onto the 3D view.
    3. Follow Camera defaults to off, so the map does not track the camera until
       the user ticks it.

    The frame is now uploaded to a GlTexture and drawn into a second Pangolin
    view, and both views sit in a LayoutEqualHorizontal container.

    The sign of the aspect matters (View::Resize in
    thirdparty/Pangolin/components/pango_display/src/view.cpp:75): a POSITIVE
    aspect fits the view inside its slot (letterbox), a NEGATIVE one overfits
    and crops. The 3D view keeps ORB-SLAM3's negative aspect so it still fills
    its slot; the frame gets a positive one so no part of the image is cut off.
    """
    if path.name != "Viewer.cpp":
        return text, 0

    n = 0

    # --- one window, sized for a side-by-side layout ------------------------
    old = '    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer",1024,768);'
    new = (
        '    // One window now holds both the 3D map and the tracked frame;\n'
        '    // 1600x900 leaves each a usable slot beside the menu panel.\n'
        '    const int kWindowWidth = 1600, kWindowHeight = 900;\n'
        '    pangolin::CreateWindowAndBind("ORB-SLAM3: Viewer",kWindowWidth,kWindowHeight);\n'
        '\n'
        '    // Pangolin sizes its root view ONLY from an X11 ConfigureNotify\n'
        '    // (thirdparty/Pangolin/components/pango_windowing/src/display_x11.cpp:422).\n'
        '    // With no window manager -- Xvfb inside a container -- a window created\n'
        '    // at its final size never receives one, so the root view stays 0x0, every\n'
        '    // child view is empty and the window renders blank.\n'
        '    //\n'
        '    // Upstream only escaped this by accident: opening the separate\n'
        '    // cv::imshow window restacked the display, and the resulting\n'
        '    // ConfigureNotify is what sized Pangolin. Folding that window into this\n'
        '    // one removed the accident, so the root view is now sized explicitly.\n'
        '    pangolin::process::Resize(kWindowWidth, kWindowHeight);'
    )
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    # --- panel wide enough for its longest label ----------------------------
    old = '    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));'
    new = (
        '    // Pangolin does not clip panel text -- Panel::Render disables the\n'
        '    // scissor test outright (pango_display/src/widgets.cpp:269) -- so a\n'
        '    // panel narrower than its longest label spills onto the 3D view.\n'
        '    //\n'
        '    // A checkbox label starts 28 px in (6 px panel inset + 18 px box +\n'
        '    // 4 px gap) and the default font, AnonymousPro at 18 px, is\n'
        '    // monospaced at 9.826 px per character. "Show Inertial Graph" is 19\n'
        '    // characters, so it ends at 28 + 186.7 = 214.7 px -- 39.7 px past the\n'
        '    // 175 px panel ORB-SLAM3 asked for. 221 px is the minimum that keeps\n'
        '    // the same 6 px margin on the right; 240 leaves room for a 20th.\n'
        '    const int kMenuPanelWidth = 240;\n'
        '    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(kMenuPanelWidth));'
    )
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    # --- follow the camera by default ---------------------------------------
    old = '    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);'
    new = '    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);'
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    # --- two views side by side ---------------------------------------------
    old = (
        '    // Add named OpenGL viewport to window and provide 3D Handler\n'
        '    pangolin::View& d_cam = pangolin::CreateDisplay()\n'
        '            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)\n'
        '            .SetHandler(new pangolin::Handler3D(s_cam));'
    )
    new = (
        '    // Both views are top-level displays with explicit bounds, the way\n'
        '    // ORB-SLAM3 already placed its single one. A LayoutEqualHorizontal\n'
        '    // container with AddDisplay() also reads well but renders nothing\n'
        '    // here, so the arrangement stays explicit.\n'
        '    //\n'
        '    // The aspect sign matters: POSITIVE fits the view inside its bounds\n'
        '    // (letterbox), NEGATIVE overfits and grows past them. ORB-SLAM3 used\n'
        '    // a negative aspect, which was harmless when the 3D view owned the\n'
        '    // whole window but covers its neighbours once it shares one.\n'
        '    // (View::Resize, thirdparty/Pangolin/components/pango_display/src/view.cpp:75)\n'
        '    const double kMapViewRight = 0.58;\n'
        '\n'
        '    pangolin::View& d_cam = pangolin::CreateDisplay()\n'
        '            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(kMenuPanelWidth),\n'
        '                       kMapViewRight, 1024.0f/768.0f)\n'
        '            .SetHandler(new pangolin::Handler3D(s_cam));\n'
        '\n'
        '    // The tracked frame, which used to be a separate cv::imshow window.\n'
        '    // Its aspect comes from the first frame, since it depends on the\n'
        '    // sensor and on whether the right image is concatenated.\n'
        '    pangolin::View& d_img = pangolin::CreateDisplay()\n'
        '            .SetBounds(0.0, 1.0, kMapViewRight, 1.0);\n'
        '\n'
        '    pangolin::GlTexture imageTexture;\n'
        '    int nLastImageCols = 0, nLastImageRows = 0;'
    )
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    # --- drop the highgui window --------------------------------------------
    old = '    cv::namedWindow("ORB-SLAM3: Current Frame");\n'
    if old in text:
        text = text.replace(old, '', 1)
        n += 1

    # --- draw the frame into the window, then swap --------------------------
    old = (
        '        pangolin::FinishFrame();\n'
        '\n'
        '        cv::Mat toShow;\n'
        '        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);\n'
        '\n'
        '        if(both){\n'
        '            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);\n'
        '            cv::hconcat(im,imRight,toShow);\n'
        '        }\n'
        '        else{\n'
        '            toShow = im;\n'
        '        }\n'
        '\n'
        '        if(mImageViewerScale != 1.f)\n'
        '        {\n'
        '            int width = toShow.cols * mImageViewerScale;\n'
        '            int height = toShow.rows * mImageViewerScale;\n'
        '            cv::resize(toShow, toShow, cv::Size(width, height));\n'
        '        }\n'
        '\n'
        '        cv::imshow("ORB-SLAM3: Current Frame",toShow);\n'
        '        cv::waitKey(mT);\n'
    )
    new = (
        '        // The frame is drawn before FinishFrame(), which swaps buffers.\n'
        '        cv::Mat toShow;\n'
        '        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);\n'
        '\n'
        '        if(both){\n'
        '            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);\n'
        '            cv::hconcat(im,imRight,toShow);\n'
        '        }\n'
        '        else{\n'
        '            toShow = im;\n'
        '        }\n'
        '\n'
        '        // This used to shrink the highgui window\'s contents. The frame\n'
        '        // now scales with its view, so the setting only picks the\n'
        '        // texture resolution.\n'
        '        if(mImageViewerScale != 1.f)\n'
        '        {\n'
        '            int width = toShow.cols * mImageViewerScale;\n'
        '            int height = toShow.rows * mImageViewerScale;\n'
        '            cv::resize(toShow, toShow, cv::Size(width, height));\n'
        '        }\n'
        '\n'
        '        if(!toShow.empty())\n'
        '        {\n'
        '            if(toShow.cols != nLastImageCols || toShow.rows != nLastImageRows)\n'
        '            {\n'
        '                // The frame changes size when the right image is\n'
        '                // concatenated, so the texture and the view\'s aspect\n'
        '                // both follow it.\n'
        '                imageTexture.Reinitialise(toShow.cols, toShow.rows, GL_RGB8,\n'
        '                                          false, 0, GL_BGR, GL_UNSIGNED_BYTE);\n'
        '                d_img.SetAspect(static_cast<double>(toShow.cols) / toShow.rows);\n'
        '                nLastImageCols = toShow.cols;\n'
        '                nLastImageRows = toShow.rows;\n'
        '            }\n'
        '\n'
        '            // Upload reads rows*cols*3 bytes contiguously, so a Mat that\n'
        '            // is a view into a larger buffer has to be compacted first.\n'
        '            const cv::Mat contiguous = toShow.isContinuous() ? toShow : toShow.clone();\n'
        '            imageTexture.Upload(contiguous.data, GL_BGR, GL_UNSIGNED_BYTE);\n'
        '\n'
        '            d_img.Activate();\n'
        '            glColor3f(1.0f,1.0f,1.0f);\n'
        '            // cv::Mat rows run top-down; OpenGL texture rows bottom-up.\n'
        '            imageTexture.RenderToViewportFlipY();\n'
        '        }\n'
        '\n'
        '        pangolin::FinishFrame();\n'
        '\n'
        '        // cv::waitKey(mT) used to pace this loop as well as pump the\n'
        '        // highgui event queue. FinishFrame() does not sleep, so the\n'
        '        // pacing is explicit now.\n'
        '        std::this_thread::sleep_for(\n'
        '            std::chrono::milliseconds(static_cast<int>(mT)));\n'
    )
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    if n and '#include <thread>' not in text:
        text = text.replace('#include <mutex>',
                            '#include <chrono>\n#include <mutex>\n#include <thread>', 1)
    # pangolin.h does not pull in process.h, which declares process::Resize.
    if n and 'display/process.h' not in text:
        text = text.replace('#include <pangolin/pangolin.h>',
                            '#include <pangolin/pangolin.h>\n'
                            '#include <pangolin/display/process.h>', 1)

    return text, n


# ---------------------------------------------------------------------------
def fix_unmatched_glend(text, path):
    """Remove the stray glEnd() in MapDrawer::DrawKeyFrames.

    The keyframe loop opens its primitive in one of two mutually exclusive
    branches -- a thicker red frustum for the map's first keyframe, a normal one
    otherwise -- so exactly ONE glBegin(GL_LINES) runs per keyframe:

        if(!pKF->GetParent()) { glLineWidth(...*5); glColor3f(1,0,0); glBegin(GL_LINES); }
        else                  { glLineWidth(...);   glColor3f(...);   glBegin(GL_LINES); }

    but it then calls glEnd() TWICE, unconditionally. glEnd() without a matching
    glBegin() is GL_INVALID_OPERATION, raised once per keyframe per rendered
    frame -- roughly 1,400 errors in 20 seconds on EuRoC MH01.

    Nothing in ORB-SLAM3 ever calls glGetError(), so the flag just accumulated
    and the bug stayed invisible. It surfaced here only because drawing the
    tracked frame through Pangolin's GlTexture brought a CheckGlDieOnError()
    into the render loop, which reports whatever error is already pending.

    The same if/else-then-single-glEnd shape appears twice more in the file and
    is correct there; only this one has the extra call.
    """
    if path.name != "MapDrawer.cpp":
        return text, 0

    old = (
        "            glEnd();\n"
        "\n"
        "            glPopMatrix();\n"
        "\n"
        "            glEnd();\n"
    )
    new = (
        "            glEnd();\n"
        "\n"
        "            glPopMatrix();\n"
    )
    if old not in text:
        return text, 0
    return text.replace(old, new, 1), 1


# ---------------------------------------------------------------------------
def fix_rectified_stereo_crash(text, path):
    """Stop Settings::operator<< dereferencing a camera that does not exist.

    A pre-rectified stereo rig has ONE calibration plus a baseline -- there is
    no second camera to describe. Settings::readCamera2 reflects that: it fills
    calibration2_/originalCalib2_ for cameraType_ PinHole and KannalaBrandt8,
    but for Rectified it only reads Stereo.b and leaves both untouched.

    operator<< then does this for any stereo sensor, unconditionally:

        for(size_t i = 0; i < settings.originalCalib2_->size(); i++)

    and the Settings constructor's init list covers only the four bool flags,
    so originalCalib2_ is not merely null -- it is uninitialised. The
    dereference is a wild pointer read and the process dies with SIGSEGV
    inside the System constructor, before a single frame is read.

    That makes every pre-rectified stereo configuration unusable, which is
    exactly what ORB-SLAM3's own KITTI stereo settings files are
    (Camera.type: "Rectified"): all of Examples/Stereo/KITTI*.yaml crash on
    startup. Confirmed under gdb on sequences 03-10.

    Two changes: the pointers start as nullptr, and the printer reports the
    baseline for a rectified rig instead of a camera that was never read.
    """
    if path.name != "Settings.cpp":
        return text, 0

    n = 0

    old = (
        "    Settings::Settings(const std::string &configFile, const int& sensor) :\n"
        "    bNeedToUndistort_(false), bNeedToRectify_(false), bNeedToResize1_(false), bNeedToResize2_(false) {"
    )
    new = (
        "    Settings::Settings(const std::string &configFile, const int& sensor) :\n"
        "    calibration1_(nullptr), calibration2_(nullptr),\n"
        "    originalCalib1_(nullptr), originalCalib2_(nullptr),\n"
        "    bNeedToUndistort_(false), bNeedToRectify_(false), bNeedToResize1_(false), bNeedToResize2_(false) {"
    )
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    old = (
        "        if(settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO){\n"
        "            output << \"\\t-Camera 2 parameters (\";\n"
        "            if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){\n"
        "                output << \"Pinhole\";\n"
        "            }\n"
        "            else{\n"
        "                output << \"Kannala-Brandt\";\n"
        "            }\n"
        "            output << \"\" << \": [\";\n"
        "            for(size_t i = 0; i < settings.originalCalib2_->size(); i++){\n"
        "                output << \" \" << settings.originalCalib2_->getParameter(i);\n"
        "            }\n"
        "            output << \" ]\" << std::endl;\n"
    )
    new = (
        "        if(settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO){\n"
        "            // A rectified rig has no second calibration -- readCamera2\n"
        "            // reads only Stereo.b for it -- so there is nothing to print\n"
        "            // here but the baseline.\n"
        "            if(settings.originalCalib2_ == nullptr){\n"
        "                output << \"\\t-Camera 2: rectified rig, baseline \"\n"
        "                       << settings.b_ << \" m\" << std::endl;\n"
        "            }\n"
        "            else{\n"
        "            output << \"\\t-Camera 2 parameters (\";\n"
        "            if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){\n"
        "                output << \"Pinhole\";\n"
        "            }\n"
        "            else{\n"
        "                output << \"Kannala-Brandt\";\n"
        "            }\n"
        "            output << \"\" << \": [\";\n"
        "            for(size_t i = 0; i < settings.originalCalib2_->size(); i++){\n"
        "                output << \" \" << settings.originalCalib2_->getParameter(i);\n"
        "            }\n"
        "            output << \" ]\" << std::endl;\n"
        "            }\n"
    )
    if old in text:
        text = text.replace(old, new, 1)
        n += 1

    return text, n


# ---------------------------------------------------------------------------
def fix_kitti_stereo_viewpoint(text, path):
    """Make Viewer.ViewpointY a real number in the KITTI 00-02 stereo settings.

    Settings::readParameter demands a real for this field and calls exit(-1)
    with "Viewer.ViewpointY parameter must be a real number, aborting..." when
    it finds an integer. Examples/Stereo/KITTI00-02.yaml writes

        Viewer.ViewpointY: -100

    where every other settings file writes a decimal (the monocular KITTI file
    has -10.0). So sequences 00-02 refuse to start in stereo, for a reason
    entirely separate from the rectified-stereo crash.
    """
    if path.name != "KITTI00-02.yaml" or "Stereo" not in str(path):
        return text, 0
    old = "Viewer.ViewpointY: -100\n"
    new = "Viewer.ViewpointY: -100.0\n"
    if old not in text:
        return text, 0
    return text.replace(old, new, 1), 1


FIXES = [
    ("COMPILEDWITHC11 guards", drop_compiledwithc11_guards),
    ("viewer single-window layout", fix_viewer_layout),
    ("unmatched glEnd in MapDrawer", fix_unmatched_glend),
    ("rectified-stereo startup crash", fix_rectified_stereo_crash),
    ("KITTI00-02 stereo ViewpointY", fix_kitti_stereo_viewpoint),
    ("viewer env switch", make_viewer_switchable),
    ("g2o solver ownership", fix_g2o_solver_ownership),
    ("mnFullBAIdx bool -> int", fix_full_ba_generation_counter),
    ("DBoW2 serialization include", add_dbow2_serialization),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    totals = {name: 0 for name, _ in FIXES}
    touched = set()

    for target in ("include", "src", "Examples"):
        for f in sorted((ROOT / target).rglob("*")):
            if not f.is_file() or f.suffix not in {".h", ".hpp", ".cpp", ".yaml"}:
                continue
            original = f.read_text(encoding="utf-8", errors="replace")
            text = original
            for name, fn in FIXES:
                text, n = fn(text, f)
                totals[name] += n
            if text != original:
                touched.add(f.relative_to(ROOT))
                if not args.check:
                    f.write_text(text, encoding="utf-8")

    for name, n in totals.items():
        print(f"  {name:<32} {n}")
    print(f"\nfiles touched: {len(touched)}")
    for t in sorted(touched):
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
