#pragma once

// =============================================================================
// DBoW2 Wrapper — Bridges upstream DBoW2 with ORB-SLAM3 requirements
// Upstream DBoW2 lacks: text vocabulary loading, boost serialization
// =============================================================================

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

namespace slam3 { namespace dbow2 {

// ============================================================================
// Type aliases
// ============================================================================
using WordId    = DBoW2::WordId;
using NodeId    = DBoW2::NodeId;
using WordValue = DBoW2::WordValue;
using BowVector     = DBoW2::BowVector;
using FeatureVector = DBoW2::FeatureVector;

// ============================================================================
// ORBVocabulary — extends TemplatedVocabulary with text file loading
// (upstream DBoW2 only supports YAML via cv::FileStorage)
// ============================================================================
class ORBVocabulary : public DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
{
public:
    using Base = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;
    using Base::Base;

    bool loadFromTextFile(const std::string &filename);
};

// ============================================================================
// Descriptor utilities
// ============================================================================

// Convert N×32 cv::Mat descriptor matrix to DBoW2 format
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &descriptors);

// Hamming distance between two ORB descriptors
int descriptorDistance(const cv::Mat &a, const cv::Mat &b);

}} // namespace slam3::dbow2

// ============================================================================
// Non-intrusive boost serialization for upstream DBoW2 types
// ============================================================================
namespace boost { namespace serialization {

template<class Archive>
void serialize(Archive &ar, DBoW2::BowVector &v, const unsigned int /*version*/)
{
    ar & static_cast<std::map<DBoW2::WordId, DBoW2::WordValue>&>(v);
}

template<class Archive>
void serialize(Archive &ar, DBoW2::FeatureVector &v, const unsigned int /*version*/)
{
    ar & static_cast<std::map<DBoW2::NodeId, std::vector<unsigned int>>&>(v);
}

}} // namespace boost::serialization
