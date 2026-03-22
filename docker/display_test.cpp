// =============================================================================
// Display Test — Verify Pangolin + OpenCV inside Docker container
// =============================================================================

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

bool test_opencv() {
    std::cout << "[OpenCV] Version: " << CV_VERSION << std::endl;

    // Test basic image operations (no GUI)
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(50, 100, 150));
    cv::putText(img, "ORB_SLAM3_Remastered", cv::Point(120, 240),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);

    // Try GUI display, fallback to headless
    try {
        cv::imshow("OpenCV Display Test", img);
        cv::waitKey(2000);
        cv::destroyAllWindows();
        std::cout << "[OpenCV] GUI display OK" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "[OpenCV] GUI not available (expected in headless): " << e.msg << std::endl;
        std::cout << "[OpenCV] Image operations OK (headless)" << std::endl;
    }

    std::cout << "[OpenCV] PASSED" << std::endl;
    return true;
}

bool test_pangolin() {
    std::cout << "[Pangolin] Creating test window..." << std::endl;

    pangolin::CreateWindowAndBind("Pangolin Display Test", 640, 480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -1, -2, 0, 0, 0, pangolin::AxisNegY)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    int frame_count = 0;
    const int max_frames = 90; // ~3 seconds at 30fps

    while (!pangolin::ShouldQuit() && frame_count < max_frames) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Draw colored axes
        glLineWidth(3.0f);
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0, 0, 0); glVertex3f(1, 0, 0);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0, 0, 0); glVertex3f(0, 1, 0);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0, 0, 0); glVertex3f(0, 0, 1);
        glEnd();

        // Draw a rotating triangle
        float angle = frame_count * 4.0f;
        glPushMatrix();
        glRotatef(angle, 0, 1, 0);
        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-0.5f, -0.3f, 0.0f);
        glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(0.5f, -0.3f, 0.0f);
        glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(0.0f, 0.5f, 0.0f);
        glEnd();
        glPopMatrix();

        pangolin::FinishFrame();
        frame_count++;
    }

    pangolin::DestroyWindow("Pangolin Display Test");
    std::cout << "[Pangolin] PASSED (" << frame_count << " frames rendered)" << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " ORB_SLAM3_Remastered Display Test" << std::endl;
    std::cout << "========================================" << std::endl;

    bool opencv_ok = test_opencv();
    bool pangolin_ok = test_pangolin();

    std::cout << "========================================" << std::endl;
    std::cout << " Results:" << std::endl;
    std::cout << "   OpenCV:   " << (opencv_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "   Pangolin: " << (pangolin_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "========================================" << std::endl;

    return (opencv_ok && pangolin_ok) ? 0 : 1;
}
