// =============================================================================
// orbslam3r/dbow2_ext/serialization.hpp
//
// Boost.Serialization support for BowVector and FeatureVector, added without
// touching upstream.
//
// WHAT ORB-SLAM3 CHANGED
//   v1.0's map save/load needs these two types inside a boost archive, so it
//   added an intrusive `friend class boost::serialization::access` plus a
//   private serialize() member to each -- 10 lines in BowVector.h and 10 in
//   FeatureVector.h (the entire delta for both files).
//
// WHY THIS DOES NOT NEED TO BE INTRUSIVE
//   Boost only requires an intrusive member when it must reach private state.
//   Both types are *public* std::map subclasses:
//
//     class BowVector     : public std::map<WordId, WordValue>
//     class FeatureVector : public std::map<NodeId, std::vector<unsigned int>>
//
//   so a free serialize() overload can serialize the base and reach everything
//   there is to reach.  Identical archive bytes, zero upstream edits.
//
// USAGE
//   Include this header anywhere a KeyFrame or Map is archived; it only has to
//   be visible before the archive operation is instantiated.
// =============================================================================
#pragma once

#include <map>
#include <vector>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>

namespace boost::serialization {

template <class Archive>
void serialize(Archive& ar, DBoW2::BowVector& v, const unsigned int /*version*/) {
  ar& base_object<std::map<DBoW2::WordId, DBoW2::WordValue>>(v);
}

template <class Archive>
void serialize(Archive& ar, DBoW2::FeatureVector& v,
               const unsigned int /*version*/) {
  ar& base_object<std::map<DBoW2::NodeId, std::vector<unsigned int>>>(v);
}

}  // namespace boost::serialization
