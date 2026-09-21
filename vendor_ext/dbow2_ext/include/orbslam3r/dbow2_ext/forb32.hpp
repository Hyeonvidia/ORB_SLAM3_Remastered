// =============================================================================
// orbslam3r/dbow2_ext/forb32.hpp
//
// DBoW2's descriptor policy for ORB, with the descriptor held as 32 bytes
// instead of as a cv::Mat.
//
// DBoW2::FORB's TDescriptor is a cv::Mat, and TemplatedVocabulary keeps one in
// every node of the tree. ORBvoc.txt has 1,082,073 nodes. A cv::Mat is a
// 96-byte header, and one that owns its data also owns a heap block for it and
// a UMatData record beside that: the 35 MB of descriptors in the file came to
// 161 MB of node array plus 116 MB of heap blocks plus the bookkeeping -- 505
// MB of resident memory before the first image, more than half of what a short
// sequence ever uses. With the 32 bytes inside the node, the array is 91 MB and
// there is nothing else.
//
// Nothing the vocabulary computes changes. TemplatedVocabulary only ever asks
// the policy for a Hamming distance, which is an exact integer however it is
// counted, so transform() descends the same branches to the same words and
// returns the same BowVector and FeatureVector bit for bit; vendor_ext's tests
// hold the two policies against each other. meanValue, toString and fromString
// follow upstream's FORB line by line (the bit-majority rule included), so that
// create() and the text format are the same too.
//
// The 32 bytes are four uint64_t so that the distance is four XORs and four
// popcounts with the alignment guaranteed. Byte order in memory is the file's
// order, exactly as in a CV_8U row.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace orbslam3r::dbow2_ext
{

    class FORB32
    {
    public:
        struct TDescriptor
        {
            std::uint64_t w[4];

            unsigned char* bytes() { return reinterpret_cast<unsigned char*>(w); }
            const unsigned char* bytes() const { return reinterpret_cast<const unsigned char*>(w); }
        };
        typedef const TDescriptor* pDescriptor;

        static const int L = 32; // descriptor length, in bytes

        static void meanValue(const std::vector<pDescriptor> &descriptors, TDescriptor &mean)
        {
            std::memset(mean.w, 0, sizeof(mean.w));
            if(descriptors.empty())
                return;
            if(descriptors.size() == 1)
            {
                mean = *descriptors[0];
                return;
            }

            std::vector<int> sum(L * 8, 0);
            for(const pDescriptor d : descriptors)
            {
                const unsigned char* p = d->bytes();
                for(int j = 0; j < L; ++j, ++p)
                    for(int bit = 0; bit < 8; ++bit)
                        if(*p & (1 << (7 - bit)))
                            ++sum[j * 8 + bit];
            }

            const int N2 = static_cast<int>(descriptors.size()) / 2 + static_cast<int>(descriptors.size() % 2);
            unsigned char* p = mean.bytes();
            for(std::size_t i = 0; i < sum.size(); ++i)
            {
                if(sum[i] >= N2)
                    *p |= static_cast<unsigned char>(1 << (7 - (i % 8)));
                if(i % 8 == 7)
                    ++p;
            }
        }

        static double distance(const TDescriptor &a, const TDescriptor &b)
        {
            return static_cast<double>(__builtin_popcountll(a.w[0] ^ b.w[0]) + __builtin_popcountll(a.w[1] ^ b.w[1]) +
                                       __builtin_popcountll(a.w[2] ^ b.w[2]) + __builtin_popcountll(a.w[3] ^ b.w[3]));
        }

        static std::string toString(const TDescriptor &a)
        {
            std::stringstream ss;
            const unsigned char* p = a.bytes();
            for(int i = 0; i < L; ++i, ++p)
                ss << static_cast<int>(*p) << " ";
            return ss.str();
        }

        static void fromString(TDescriptor &a, const std::string &s)
        {
            std::memset(a.w, 0, sizeof(a.w));
            unsigned char* p = a.bytes();
            std::stringstream ss(s);
            for(int i = 0; i < L; ++i, ++p)
            {
                int n;
                ss >> n;
                if(!ss.fail())
                    *p = static_cast<unsigned char>(n);
            }
        }

        // The rows of an N x 32 CV_8U matrix -- what ORBextractor produces -- as
        // descriptors. One copy of N * 32 bytes, where building a cv::Mat header
        // per row, as DBoW2::FORB needs, is N reference-counted objects.
        static std::vector<TDescriptor> FromMat(const cv::Mat &descriptors)
        {
            CV_Assert(descriptors.empty() || (descriptors.type() == CV_8U && descriptors.cols == L));
            std::vector<TDescriptor> out(static_cast<std::size_t>(descriptors.rows));
            for(int i = 0; i < descriptors.rows; ++i)
                std::memcpy(out[static_cast<std::size_t>(i)].w, descriptors.ptr<unsigned char>(i), L);
            return out;
        }
    };

} // namespace orbslam3r::dbow2_ext
