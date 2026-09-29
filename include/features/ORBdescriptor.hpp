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

#ifndef ORBDESCRIPTOR_H
#define ORBDESCRIPTOR_H

#include <cstdint>
#include <cstring>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

    // The part of ORB matching that MapPoint and Frame use: comparing two
    // descriptors. It lives apart from ORBmatcher, which works on Frames,
    // KeyFrames and MapPoints, so that those classes can use it without
    // depending on the matcher.
    class ORBdescriptor
    {
    public:
        // The Hamming distance between two ORB descriptors of 32 bytes. Every
        // search compares descriptors in its innermost loop, so it is here to
        // be inlined, and counts 64 bits at a time: one instruction where the
        // processor has it, as every ARMv8 does.
        static int Distance(const unsigned char* a, const unsigned char* b)
        {
            std::uint64_t x[4], y[4];
            std::memcpy(x, a, 32);
            std::memcpy(y, b, 32);
            int dist = 0;
            for(int i = 0; i < 4; ++i)
                dist += Count(x[i] ^ y[i]);
            return dist;
        }

        static int Distance(const cv::Mat &a, const cv::Mat &b) { return Distance(a.data, b.data); }

        // A descriptor by value, for what hands one out under a lock: copying
        // 32 bytes allocates nothing, a cv::Mat of them does.
        struct Bytes
        {
            unsigned char b[32];
            operator const unsigned char*() const { return b; }
        };

        static constexpr int TH_LOW = 50;
        static constexpr int TH_HIGH = 100;

    private:
        static int Count(std::uint64_t v)
        {
#if defined(__aarch64__) || defined(__POPCNT__)
            return __builtin_popcountll(v);
#else
            // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
            v = v - ((v >> 1) & 0x5555555555555555ULL);
            v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
            return static_cast<int>((((v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
        }
    };

} // namespace ORB_SLAM3

#endif // ORBDESCRIPTOR_H
