// The words of an image in one block (atlas/BowWords.hpp) must score exactly
// as DBoW2 scores the same words in its std::map, and the flat map must find,
// insert and erase as a std::map does.
#include <cstdio>
#include <map>
#include <random>

#include <DBoW2/BowVector.h>

#include "atlas/BowWords.hpp"
#include "common/FlatMap.hpp"

int main()
{
    int rc = 0;
    const auto expect = [&rc](bool ok, const char* what)
    {
        if(!ok)
        {
            std::fprintf(stderr, "FAILED: %s\n", what);
            rc = 1;
        }
    };

    std::mt19937 rng(11);
    const ORB_SLAM3::ORBVocabulary voc(10, 6, DBoW2::TF_IDF, DBoW2::L1_NORM);
    long nDifferent = 0;
    for(int n = 0; n < 2000; ++n)
    {
        // Two images of a place share words; two of different places few.
        const unsigned int nWords = 1 + rng() % 2000, range = (n % 2) ? 4000 : 1000000;
        DBoW2::BowVector a, b;
        for(unsigned int i = 0; i < nWords; ++i)
        {
            a.addWeight(rng() % range, (rng() % 1000) / 1000.0);
            b.addWeight(rng() % range, (rng() % 1000) / 1000.0);
        }
        a.normalize(DBoW2::L1);
        b.normalize(DBoW2::L1);
        ORB_SLAM3::BowWords fa, fb;
        fa.assign(a);
        fb.assign(b);
        nDifferent += ORB_SLAM3::Score(voc, fa, fb) != voc.score(a, b);
        nDifferent += ORB_SLAM3::Score(voc, fa, fa) != voc.score(a, a);
        nDifferent += fa.size() != a.size() || fa.capacity() != a.size();
    }
    expect(nDifferent == 0, "the score of the words in one block is DBoW2's score of them");

    std::map<long, int> reference;
    ORB_SLAM3::FlatMap<long, int> flat;
    long nWrong = 0;
    for(int n = 0; n < 200000; ++n)
    {
        const long key = rng() % 64;
        switch(rng() % 4)
        {
            case 0:
                reference[key] = n;
                flat[key] = n;
                break;
            case 1:
                nWrong += reference.erase(key) != flat.erase(key);
                break;
            case 2:
                nWrong += reference.count(key) != flat.count(key);
                if(reference.count(key))
                    nWrong += reference.find(key)->second != flat.find(key)->second;
                break;
            default:
            {
                nWrong += reference.size() != flat.size();
                std::map<long, int>::const_iterator it = reference.begin();
                for(const std::pair<long, int> &entry : flat)
                {
                    nWrong += it == reference.end() || it->first != entry.first || it->second != entry.second;
                    if(it != reference.end())
                        ++it;
                }
                const bool bBoth = reference.lower_bound(key) == reference.end();
                nWrong += bBoth != (flat.lower_bound(key) == flat.end());
                if(!bBoth)
                    nWrong += reference.lower_bound(key)->first != flat.lower_bound(key)->first;
            }
        }
    }
    expect(nWrong == 0, "a flat map holds what a std::map holds, in its order");

    std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
