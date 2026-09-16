// Smallest program that answers "can Pangolin open a usable window on THIS
// display?" -- used to test X11 forwarding without paying ORB-SLAM3's
// vocabulary load first.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <pangolin/pangolin.h>
#include <pangolin/display/process.h>

int main(int argc, char** argv) {
  const int seconds = argc > 1 ? std::atoi(argv[1]) : 6;
  const int W = 640, H = 480;

  std::printf("DISPLAY=%s  PANGOLIN_WINDOW_URI=%s\n",
              std::getenv("DISPLAY") ? std::getenv("DISPLAY") : "(unset)",
              std::getenv("PANGOLIN_WINDOW_URI") ? std::getenv("PANGOLIN_WINDOW_URI") : "(unset)");
  try {
    pangolin::CreateWindowAndBind("GLProbe", W, H);
  } catch (const std::exception& e) {
    std::printf("RESULT: window creation threw: %s\n", e.what());
    return 1;
  }
  pangolin::process::Resize(W, H);

  std::printf("GL_VENDOR   : %s\n", glGetString(GL_VENDOR));
  std::printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));
  std::printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));

  pangolin::View& d = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0);
  int frames = 0;
  bool reported = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  while (std::chrono::steady_clock::now() < deadline && !pangolin::ShouldQuit()) {
    glClearColor(0.1f, 0.4f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d.Activate();
    glColor3f(1.0f, 1.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    glVertex3f(-0.6f, -0.6f, 0.0f);
    glVertex3f(0.6f, -0.6f, 0.0f);
    glVertex3f(0.0f, 0.6f, 0.0f);
    glEnd();
    if (frames == 5 && !reported) {
      reported = true;
      const pangolin::Viewport& b = pangolin::DisplayBase().v;
      std::printf("base viewport: l=%d b=%d w=%d h=%d   view: w=%d h=%d\n",
                  b.l, b.b, b.w, b.h, d.v.w, d.v.h);
      unsigned char px[4] = {0, 0, 0, 0};
      glReadPixels(b.w / 2, b.h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
      std::printf("back-buffer centre: %d %d %d\n", px[0], px[1], px[2]);
    }
    pangolin::FinishFrame();
    ++frames;
    std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }
  GLenum err = glGetError();
  std::printf("RESULT: %d frames drawn, last GL error 0x%04X\n", frames, err);
  return 0;
}
