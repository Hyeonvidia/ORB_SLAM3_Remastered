// =============================================================================
// The vocabulary tree in five arrays instead of a million nodes.
//
// DBoW2::TemplatedVocabulary keeps a Node for every node of the tree: its id,
// its weight, a vector of its children, its parent, its descriptor and its
// word. For ORBvoc.txt -- 1,111,111 nodes of 32-byte descriptors -- that is
// 106 MB: 84 bytes a node, and a heap block for the children of each inner
// one. It is the largest thing in the process when the map is a room.
//
// What the system asks of a vocabulary is to turn the descriptors of an image
// into its words (transform) and to score two images' words (score). Neither
// needs the parent, nor a vector to find ten children. Here the nodes are in
// the order in which the tree is walked breadth first, so that the children of
// a node are next to each other:
//
//   descriptor   32 bytes    compared against a feature on the way down
//   first child   4          where its children begin; they are contiguous
//   children      1          how many
//   node id       4          the node's number in the file, which is what a
//                            FeatureVector is keyed by, in memory and on disk
//   word          4          the word's number, for a leaf
//   weight        8          as a double, as DBoW2 has it
//
// 53 bytes a node, 59 MB, in six blocks. The ten descriptors a feature is
// compared against at each level are 320 consecutive bytes.
//
// The words come out through DBoW2's own BowVector and FeatureVector, and the
// score through DBoW2's own scoring object, so what is computed is what
// TextFileVocabulary<FORB32> computes from the same file, bit for bit;
// tests/test_bowwords holds the two against each other.
//
// Not here: create(), save, the single-feature and stop-word interfaces. The
// system uses none of them; TextFileVocabulary is there for who does.
// =============================================================================
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <DBoW2/ScoringObject.h>

#include "orbslam3r/dbow2_ext/forb32.hpp"

namespace orbslam3r::dbow2_ext
{

    class CompactVocabulary
    {
    public:
        typedef FORB32::TDescriptor TDescriptor;

        explicit CompactVocabulary(int k = 10, int L = 5, DBoW2::WeightingType weighting = DBoW2::TF_IDF,
                                   DBoW2::ScoringType scoring = DBoW2::L1_NORM)
            : mk(k), mL(L), mWeighting(weighting), mScoring(scoring)
        {
            CreateScoringObject();
        }

        // The text format of ORB-SLAM3's ORBvoc.txt: "k L scoring weighting",
        // then a line for every node but the root, "parent leaf d0 ... d31
        // weight", a node's number being that of its line.
        bool loadFromTextFile(const std::string &filename);

        unsigned int size() const { return mnWords; }
        bool empty() const { return mnWords == 0; }
        DBoW2::ScoringType getScoringType() const { return mScoring; }
        DBoW2::WeightingType getWeightingType() const { return mWeighting; }

        // As DBoW2::TemplatedVocabulary::transform with four arguments.
        void transform(const std::vector<TDescriptor> &features, DBoW2::BowVector &v, DBoW2::FeatureVector &fv,
                       int levelsup) const;

        double score(const DBoW2::BowVector &a, const DBoW2::BowVector &b) const { return mpScoring->score(a, b); }

        std::map<std::string, std::size_t> MemoryFootprint() const
        {
            auto Chunk = [](std::size_t n) -> std::size_t
            { return n == 0 ? 0 : std::max<std::size_t>(32, (n + 8 + 15) / 16 * 16); };
            std::map<std::string, std::size_t> f;
            f["node descriptors"] = Chunk(mvDescriptors.capacity() * sizeof(TDescriptor));
            f["node children, where and how many"] = Chunk(mvFirstChild.capacity() * sizeof(std::uint32_t)) +
                                                     Chunk(mvChildren.capacity());
            f["node ids and words"] = Chunk(mvNodeId.capacity() * sizeof(DBoW2::NodeId)) +
                                      Chunk(mvWord.capacity() * sizeof(DBoW2::WordId));
            f["node weights"] = Chunk(mvWeight.capacity() * sizeof(DBoW2::WordValue));
            return f;
        }

    private:
        void CreateScoringObject()
        {
            switch(mScoring)
            {
                case DBoW2::L1_NORM:
                    mpScoring = std::make_unique<DBoW2::L1Scoring>();
                    break;
                case DBoW2::L2_NORM:
                    mpScoring = std::make_unique<DBoW2::L2Scoring>();
                    break;
                case DBoW2::CHI_SQUARE:
                    mpScoring = std::make_unique<DBoW2::ChiSquareScoring>();
                    break;
                case DBoW2::KL:
                    mpScoring = std::make_unique<DBoW2::KLScoring>();
                    break;
                case DBoW2::BHATTACHARYYA:
                    mpScoring = std::make_unique<DBoW2::BhattacharyyaScoring>();
                    break;
                case DBoW2::DOT_PRODUCT:
                    mpScoring = std::make_unique<DBoW2::DotProductScoring>();
                    break;
            }
        }

        // Down the tree from the root, at each node to the child nearest the
        // feature and to the first of them if several are: the word at the
        // bottom, its weight, and the node passed levelsup above it.
        void Word(const TDescriptor &feature, DBoW2::WordId &word, DBoW2::WordValue &weight, DBoW2::NodeId &nid,
                  int levelsup) const
        {
            const int nidLevel = mL - levelsup;
            nid = 0;
            std::uint32_t at = 0;
            int level = 0;
            do
            {
                ++level;
                const std::uint32_t first = mvFirstChild[at];
                const std::uint32_t end = first + mvChildren[at];
                double best = std::numeric_limits<double>::max();
                for(std::uint32_t child = first; child < end; ++child)
                {
                    const double d = FORB32::distance(feature, mvDescriptors[child]);
                    if(d < best)
                    {
                        best = d;
                        at = child;
                    }
                }
                if(level == nidLevel)
                    nid = mvNodeId[at];
            } while(mvChildren[at] != 0);
            word = mvWord[at];
            weight = mvWeight[at];
        }

        int mk;
        int mL;
        DBoW2::WeightingType mWeighting;
        DBoW2::ScoringType mScoring;
        std::unique_ptr<DBoW2::GeneralScoring> mpScoring;

        // A node is an index into each of these; the root is 0.
        std::vector<TDescriptor> mvDescriptors;
        std::vector<std::uint32_t> mvFirstChild;
        std::vector<std::uint8_t> mvChildren;
        std::vector<DBoW2::NodeId> mvNodeId;
        std::vector<DBoW2::WordId> mvWord;
        std::vector<DBoW2::WordValue> mvWeight;
        unsigned int mnWords = 0;
    };

    inline void CompactVocabulary::transform(const std::vector<TDescriptor> &features, DBoW2::BowVector &v,
                                             DBoW2::FeatureVector &fv, int levelsup) const
    {
        v.clear();
        fv.clear();
        if(mvDescriptors.size() < 2)
            return;

        DBoW2::LNorm norm;
        const bool bNormalize = mpScoring->mustNormalize(norm);

        unsigned int iFeature = 0;
        if(mWeighting == DBoW2::TF || mWeighting == DBoW2::TF_IDF)
        {
            for(const TDescriptor &feature : features)
            {
                DBoW2::WordId word;
                DBoW2::WordValue weight;
                DBoW2::NodeId nid;
                Word(feature, word, weight, nid, levelsup);
                if(weight > 0) // not a stop word
                {
                    v.addWeight(word, weight);
                    fv.addFeature(nid, iFeature);
                }
                ++iFeature;
            }
            if(!v.empty() && !bNormalize)
            {
                const double nd = v.size();
                for(DBoW2::BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                    vit->second /= nd;
            }
        }
        else // IDF or BINARY
        {
            for(const TDescriptor &feature : features)
            {
                DBoW2::WordId word;
                DBoW2::WordValue weight;
                DBoW2::NodeId nid;
                Word(feature, word, weight, nid, levelsup);
                if(weight > 0)
                {
                    v.addIfNotExist(word, weight);
                    fv.addFeature(nid, iFeature);
                }
                ++iFeature;
            }
        }

        if(bNormalize)
            v.normalize(norm);
    }

    inline bool CompactVocabulary::loadFromTextFile(const std::string &filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if(!f.is_open())
        {
            std::cerr << "Vocabulary loading failure: cannot open '" << filename << "'\n";
            return false;
        }
        std::string line;
        if(!std::getline(f, line))
        {
            std::cerr << "Vocabulary loading failure: '" << filename << "' is empty\n";
            return false;
        }
        int k = -1, L = -1, scoring = -1, weighting = -1;
        {
            char* p = line.data();
            k = static_cast<int>(std::strtol(p, &p, 10));
            L = static_cast<int>(std::strtol(p, &p, 10));
            scoring = static_cast<int>(std::strtol(p, &p, 10));
            weighting = static_cast<int>(std::strtol(p, &p, 10));
        }
        if(k < 2 || k > 20 || L < 1 || L > 10 || scoring < 0 || scoring > 5 || weighting < 0 || weighting > 3)
        {
            std::cerr << "Vocabulary loading failure: '" << filename << "' is not a valid vocabulary text file\n";
            return false;
        }
        mk = k;
        mL = L;
        mScoring = static_cast<DBoW2::ScoringType>(scoring);
        mWeighting = static_cast<DBoW2::WeightingType>(weighting);
        CreateScoringObject();

        // The nodes as the file has them, the root first.
        std::size_t nExpected = 1;
        for(int level = 0, n = 1; level < L; ++level)
            nExpected += (n *= k);
        std::vector<TDescriptor> descriptors(1);
        std::vector<std::uint32_t> parents(1, 0);
        std::vector<std::uint8_t> leaves(1, 0);
        std::vector<DBoW2::WordValue> weights(1, 0);
        std::memset(descriptors[0].w, 0, sizeof(descriptors[0].w));
        descriptors.reserve(nExpected);
        parents.reserve(nExpected);
        leaves.reserve(nExpected);
        weights.reserve(nExpected);

        while(std::getline(f, line))
        {
            if(line.find_first_not_of(" \t\r\n") == std::string::npos)
                continue;
            const std::size_t nid = descriptors.size();
            char* p = line.data();
            const long parent = std::strtol(p, &p, 10);
            const long leaf = std::strtol(p, &p, 10);
            if(parent < 0 || static_cast<std::size_t>(parent) >= nid)
            {
                std::cerr << "Vocabulary loading failure: node " << nid << " names parent " << parent
                          << ", which does not precede it\n";
                return false;
            }
            TDescriptor d;
            std::memset(d.w, 0, sizeof(d.w));
            for(int i = 0; i < FORB32::L; ++i)
            {
                char* q = p;
                const long byte = std::strtol(p, &q, 10);
                if(q != p)
                    d.bytes()[i] = static_cast<unsigned char>(byte);
                p = q;
            }
            descriptors.push_back(d);
            parents.push_back(static_cast<std::uint32_t>(parent));
            leaves.push_back(leaf > 0 ? 1 : 0);
            weights.push_back(std::strtod(p, &p));
        }
        const std::size_t nNodes = descriptors.size();

        // How many children each has, and the order that puts them together:
        // breadth first, the children of a node in the order of the file.
        std::vector<std::uint32_t> nChildren(nNodes, 0);
        for(std::size_t i = 1; i < nNodes; ++i)
            ++nChildren[parents[i]];
        for(std::size_t i = 0; i < nNodes; ++i)
            if(nChildren[i] > 255)
            {
                std::cerr << "Vocabulary loading failure: node " << i << " has " << nChildren[i] << " children\n";
                return false;
            }
        std::vector<std::uint32_t> firstOf(nNodes + 1, 0); // in the list of children by parent
        for(std::size_t i = 0; i < nNodes; ++i)
            firstOf[i + 1] = firstOf[i] + nChildren[i];
        std::vector<std::uint32_t> byParent(nNodes > 0 ? nNodes - 1 : 0);
        {
            std::vector<std::uint32_t> next(firstOf.begin(), firstOf.end() - 1);
            for(std::size_t i = 1; i < nNodes; ++i)
                byParent[next[parents[i]]++] = static_cast<std::uint32_t>(i);
        }
        std::vector<std::uint32_t> order; // order[place] = the node of the file that goes there
        order.reserve(nNodes);
        order.push_back(0);
        for(std::size_t place = 0; place < order.size(); ++place)
        {
            const std::uint32_t node = order[place];
            for(std::uint32_t c = firstOf[node]; c < firstOf[node + 1]; ++c)
                order.push_back(byParent[c]);
        }
        if(order.size() != nNodes)
        {
            std::cerr << "Vocabulary loading failure: " << nNodes - order.size() << " nodes are not under the root\n";
            return false;
        }

        // Words are numbered as the file has the leaves.
        std::vector<DBoW2::WordId> words(nNodes, 0);
        mnWords = 0;
        for(std::size_t i = 1; i < nNodes; ++i)
            if(leaves[i])
                words[i] = mnWords++;

        bool bInOrder = true;
        for(std::size_t place = 0; place < nNodes && bInOrder; ++place)
            bInOrder = order[place] == place;
        if(bInOrder)
        {
            // ORBvoc.txt is: nothing to move, and nothing held twice.
            mvDescriptors.swap(descriptors);
            mvWeight.swap(weights);
            mvWord.swap(words);
        }
        else
        {
            mvDescriptors.resize(nNodes);
            mvWeight.resize(nNodes);
            mvWord.resize(nNodes);
            for(std::size_t place = 0; place < nNodes; ++place)
            {
                mvDescriptors[place] = descriptors[order[place]];
                mvWeight[place] = weights[order[place]];
                mvWord[place] = words[order[place]];
            }
        }
        mvDescriptors.shrink_to_fit();
        mvWeight.shrink_to_fit();
        mvWord.shrink_to_fit();
        mvFirstChild.assign(nNodes, 0);
        mvChildren.resize(nNodes);
        mvNodeId.resize(nNodes);
        std::uint32_t nextChild = 1;
        for(std::size_t place = 0; place < nNodes; ++place)
        {
            const std::uint32_t node = order[place];
            mvChildren[place] = static_cast<std::uint8_t>(nChildren[node]);
            mvFirstChild[place] = nChildren[node] ? nextChild : 0;
            nextChild += nChildren[node];
            mvNodeId[place] = node;
        }
        return true;
    }

} // namespace orbslam3r::dbow2_ext
