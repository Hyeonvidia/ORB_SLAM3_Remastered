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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <atomic>
#include <list>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "common/WorkerPool.hpp"

namespace ORB_SLAM3
{

#ifdef ORBSLAM3R_ORB_PROFILE
    // Where an extraction's time goes, summed over the calls and over the
    // threads, in nanoseconds. tests/bench_orb compiles the extractor with
    // this; the library does not.
    struct ORBprofile
    {
        enum Phase
        {
            PYRAMID,
            FAST,
            DISTRIBUTE,
            ORIENTATION,
            BLUR,
            DESCRIPTORS,
            GATHER,
            N_PHASES
        };
        static std::atomic<long long> ns[N_PHASES];
    };
#endif

    class ExtractorNode
    {
    public:
        ExtractorNode() : bNoMore(false) {}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class ORBextractor
    {
    public:
        enum
        {
            HARRIS_SCORE = 0,
            FAST_SCORE = 1
        };

        // nThreads: how many threads share the levels of the pyramid, the
        // caller among them; 0 for DefaultThreads(). What is extracted does not
        // depend on it.
        ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST, int nThreads = 0);

        // ORBSLAM3R_ORB_THREADS if it is set, or two where there are two.
        static int DefaultThreads();

        ~ORBextractor() {}

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        int operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint> &_keypoints,
                       cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

        int inline GetLevels() { return nlevels; }

        float inline GetScaleFactor() { return scaleFactor; }

        std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

        std::vector<float> inline GetInverseScaleFactors() { return mvInvScaleFactor; }

        std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

        std::vector<float> inline GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

        std::vector<cv::Mat> mvImagePyramid;

    protected:
        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(int level, std::vector<cv::KeyPoint> &keypoints);
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                    const int &maxX, const int &minY, const int &maxY,
                                                    const int &nFeatures, const int &level);

        // The sampling pattern, as floats: they are multiplied by a sine and a
        // cosine for every descriptor.
        std::vector<float> mvPatternX;
        std::vector<float> mvPatternY;

        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        // Each level with its border, which mvImagePyramid is a window onto,
        // and each level blurred: kept, so that a frame allocates neither.
        std::vector<cv::Mat> mvBordered;
        std::vector<cv::Mat> mvBlurred;

        std::unique_ptr<WorkerPool> mpPool;
    };

} // namespace ORB_SLAM3

#endif
