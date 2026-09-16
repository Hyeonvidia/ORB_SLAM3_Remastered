// =============================================================================
// orbslam3r/dbow2_ext/text_vocabulary.hpp
//
// The plain-text vocabulary format, as a subclass instead of a fork.
//
// WHAT ORB-SLAM3 CHANGED IN DBoW2
//   Measured with tools/upstream_delta.py: 237 lines across 7 files, and 139 of
//   those lines are this one feature.  Upstream DBoW2 can only persist a
//   vocabulary through cv::FileStorage (YAML/XML), which takes minutes to parse
//   for the 971,814-word ORB vocabulary.  Raul Mur-Artal added a flat text format in
//   2015 for ORB-SLAM2 -- that is what Vocabulary/ORBvoc.txt is -- by editing
//   TemplatedVocabulary.h in place.
//
// WHY A SUBCLASS WORKS
//   Everything the loader touches (m_k, m_L, m_scoring, m_weighting, m_nodes,
//   m_words, Node, createScoringObject) is `protected` upstream, and
//   ~TemplatedVocabulary is virtual.  So the feature can be bolted on from
//   outside, and thirdparty/DBoW2 stays a byte-identical upstream checkout.
//
// FILE FORMAT
//   line 1     : k  L  scoring  weighting
//   lines 2..n : parent_id  is_leaf  d0 d1 ... d31  weight
//   Node ids are implicit: the n-th body line is node n, so a node's parent
//   always appears before it.
//
// TWO DEFECTS IN THE ORIGINAL ARE FIXED HERE (see docs/WRAPPERS.md)
//   1. A missing file was reported as success.  The original guard is
//      `if (f.eof()) return false;`, but eof() is false on a stream that failed
//      to open, so a bad path produced an empty vocabulary and a silent,
//      total loss of place recognition.  Now checked with is_open().
//   2. A trailing newline appended a garbage node.  `while (!f.eof())` runs one
//      extra iteration after the last real line, and the original body grew
//      m_nodes and read an uninitialised parent id from the empty stream.
//      Blank lines are now skipped.
// =============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace orbslam3r::dbow2_ext
{

    template<class TDescriptor, class F>
    class TextFileVocabulary : public DBoW2::TemplatedVocabulary<TDescriptor, F>
    {
        using Base = DBoW2::TemplatedVocabulary<TDescriptor, F>;

    public:
        using Base::Base;

        // Replaces the vocabulary's contents with the file's.
        // Returns false and leaves the object empty if the file is missing or its
        // header is not a plausible vocabulary header.
        bool loadFromTextFile(const std::string &filename);

        // Writes the vocabulary in the same format. Node 0 (the root) is implicit
        // and not written, matching the loader.
        bool saveToTextFile(const std::string &filename) const;

    protected:
        using Base::m_k;
        using Base::m_L;
        using Base::m_nodes;
        using Base::m_scoring;
        using Base::m_weighting;
        using Base::m_words;
        using typename Base::Node;
    };

    // ---------------------------------------------------------------------------
    template<class TDescriptor, class F>
    bool TextFileVocabulary<TDescriptor, F>::loadFromTextFile(const std::string &filename)
    {
        std::ifstream f(filename);
        // Defect 1: the original tested f.eof() here, which is false for a stream
        // that never opened, so a wrong path silently yielded an empty vocabulary.
        if(!f.is_open())
        {
            std::cerr << "Vocabulary loading failure: cannot open '" << filename << "'\n";
            return false;
        }

        m_words.clear();
        m_nodes.clear();

        std::string line;
        if(!std::getline(f, line))
        {
            std::cerr << "Vocabulary loading failure: '" << filename << "' is empty\n";
            return false;
        }

        int scoring = -1;
        int weighting = -1;
        {
            std::istringstream header(line);
            header >> m_k >> m_L >> scoring >> weighting;
        }

        // The same sanity window the original used: a branching factor and depth
        // that a real vocabulary could have, and valid enum ordinals.
        if(m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || scoring < 0 || scoring > 5 || weighting < 0 || weighting > 3)
        {
            std::cerr << "Vocabulary loading failure: '" << filename << "' is not a valid vocabulary text file\n";
            return false;
        }

        m_scoring = static_cast<DBoW2::ScoringType>(scoring);
        m_weighting = static_cast<DBoW2::WeightingType>(weighting);
        this->createScoringObject();

        const auto expected_nodes = static_cast<std::size_t>(
            (std::pow(static_cast<double>(m_k), static_cast<double>(m_L) + 1) - 1) / (m_k - 1));
        m_nodes.reserve(expected_nodes);
        m_words.reserve(static_cast<std::size_t>(std::pow(static_cast<double>(m_k), static_cast<double>(m_L) + 1)));

        m_nodes.resize(1); // the root
        m_nodes[0].id = 0;

        while(std::getline(f, line))
        {
            // Defect 2: a trailing newline used to add a node with an uninitialised
            // parent id.
            if(line.find_first_not_of(" \t\r\n") == std::string::npos)
                continue;

            std::istringstream ss(line);
            const auto nid = m_nodes.size();
            m_nodes.resize(m_nodes.size() + 1);
            m_nodes[nid].id = static_cast<DBoW2::NodeId>(nid);

            int parent_id = 0;
            int is_leaf = 0;
            ss >> parent_id >> is_leaf;
            if(parent_id < 0 || static_cast<std::size_t>(parent_id) >= nid)
            {
                std::cerr << "Vocabulary loading failure: node " << nid << " names parent " << parent_id
                          << ", which does not precede it\n";
                return false;
            }
            m_nodes[nid].parent = static_cast<DBoW2::NodeId>(parent_id);
            m_nodes[parent_id].children.push_back(static_cast<DBoW2::NodeId>(nid));

            // F::L bytes, written as decimal numbers, then reassembled by the
            // descriptor policy's own parser.
            std::ostringstream descriptor;
            for(int i = 0; i < F::L; ++i)
            {
                std::string element;
                ss >> element;
                descriptor << element << ' ';
            }
            F::fromString(m_nodes[nid].descriptor, descriptor.str());

            ss >> m_nodes[nid].weight;

            if(is_leaf > 0)
            {
                const auto wid = m_words.size();
                m_words.resize(wid + 1);
                m_nodes[nid].word_id = static_cast<DBoW2::WordId>(wid);
                m_words[wid] = &m_nodes[nid];
            }
            else
            {
                m_nodes[nid].children.reserve(m_k);
            }
        }

        return true;
    }

    // ---------------------------------------------------------------------------
    template<class TDescriptor, class F>
    bool TextFileVocabulary<TDescriptor, F>::saveToTextFile(const std::string &filename) const
    {
        std::ofstream f(filename);
        if(!f.is_open())
        {
            std::cerr << "Vocabulary save failure: cannot write '" << filename << "'\n";
            return false;
        }

        f << m_k << ' ' << m_L << ' ' << m_scoring << ' ' << m_weighting << '\n';

        for(std::size_t i = 1; i < m_nodes.size(); ++i)
        {
            const Node &node = m_nodes[i];
            f << node.parent << ' ' << (node.isLeaf() ? 1 : 0) << ' ' << F::toString(node.descriptor) << ' '
              << static_cast<double>(node.weight) << '\n';
        }

        return static_cast<bool>(f);
    }

} // namespace orbslam3r::dbow2_ext
