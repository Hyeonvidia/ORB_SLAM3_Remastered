// What ORB extraction costs on its own, and where inside it: one thread, the
// images read before the clock starts, a few seconds a run.
//
//   bench_orb <folder of images> [images=200] [features=2000] [iniThFAST=20] [minThFAST=7] [passes=5] [threads=0]
//
// The images are taken evenly from the folder. The last line is a hash of
// every keypoint and descriptor extracted, so that a change meant to leave the
// output alone can be seen to, and one that does not can be seen not to.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "features/ORBextractor.hpp"

namespace
{
    std::uint64_t Hash(std::uint64_t h, const void* p, std::size_t n)
    {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for(std::size_t i = 0; i < n; ++i)
            h = (h ^ b[i]) * 1099511628211ULL;
        return h;
    }
} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::fprintf(stderr, "usage: %s <folder of images> [images] [features] [iniThFAST] [minThFAST] [passes]\n",
                     argv[0]);
        return 2;
    }
    const std::size_t nImages = argc > 2 ? std::atoi(argv[2]) : 200;
    const int nFeatures = argc > 3 ? std::atoi(argv[3]) : 2000;
    const int iniThFAST = argc > 4 ? std::atoi(argv[4]) : 20;
    const int minThFAST = argc > 5 ? std::atoi(argv[5]) : 7;
    const int nPasses = argc > 6 ? std::atoi(argv[6]) : 5;
    const int nThreads = argc > 7 ? std::atoi(argv[7]) : 0;

    std::vector<cv::String> names;
    cv::glob(std::string(argv[1]) + "/*.png", names);
    std::sort(names.begin(), names.end());
    std::vector<cv::Mat> images;
    for(std::size_t i = 0; i < nImages && !names.empty(); ++i)
    {
        const cv::Mat image = cv::imread(names[i * names.size() / nImages], cv::IMREAD_GRAYSCALE);
        if(!image.empty())
            images.push_back(image);
    }
    if(images.empty())
    {
        std::fprintf(stderr, "no images in %s\n", argv[1]);
        return 1;
    }

    ORB_SLAM3::ORBextractor extractor(nFeatures, 1.2f, 8, iniThFAST, minThFAST, nThreads);
    std::vector<int> lapping = {0, 0};
    std::vector<double> ms;
    std::uint64_t hash = 14695981039346656037ULL;
    std::size_t nKeypoints = 0;
    for(int pass = 0; pass < nPasses; ++pass)
    {
        for(const cv::Mat &image : images)
        {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            extractor(image, cv::Mat(), keypoints, descriptors, lapping);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if(pass == 0)
            {
                nKeypoints += keypoints.size();
                for(const cv::KeyPoint &k : keypoints)
                {
                    const float v[5] = {k.pt.x, k.pt.y, k.angle, k.response, k.size};
                    hash = Hash(hash, v, sizeof(v));
                    hash = Hash(hash, &k.octave, sizeof(k.octave));
                }
                for(int r = 0; r < descriptors.rows; ++r)
                    hash = Hash(hash, descriptors.ptr(r), 32);
            }
        }
    }

    double sum = 0;
    for(double v : ms)
        sum += v;
    std::sort(ms.begin(), ms.end());
    std::printf("%zu images of %d x %d, %d passes, %d features asked for, %.0f found per image\n", images.size(),
                images[0].cols, images[0].rows, nPasses, nFeatures, double(nKeypoints) / images.size());
    std::printf("per image: mean %.3f ms, median %.3f ms, fastest tenth %.3f ms\n", sum / ms.size(),
                ms[ms.size() / 2], ms[ms.size() / 10]);
#ifdef ORBSLAM3R_ORB_PROFILE
    static const char* const names_[ORB_SLAM3::ORBprofile::N_PHASES] = {
        "pyramid", "FAST", "distribute", "orientation", "blur", "descriptors", "gather"};
    double accounted = 0;
    for(int p = 0; p < ORB_SLAM3::ORBprofile::N_PHASES; ++p)
    {
        const double each = ORB_SLAM3::ORBprofile::ns[p].load() / 1e6 / ms.size();
        accounted += each;
        std::printf("  %-12s %7.3f ms  %5.1f %%\n", names_[p], each, 100 * each * ms.size() / sum);
    }
    std::printf("  %-12s %7.3f ms  (the phases are summed over the threads)\n", "the rest",
                sum / ms.size() - accounted);
#endif
    std::printf("output %016llx\n", static_cast<unsigned long long>(hash));
    return 0;
}
