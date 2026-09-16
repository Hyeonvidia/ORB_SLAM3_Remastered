// =============================================================================
// orbslam3r/dbow2_ext/orb_vocabulary.hpp
//
// The vocabulary type the SLAM system uses: upstream DBoW2's ORB descriptor
// policy, plus the text-file format from text_vocabulary.hpp.
//
// Replaces ORB-SLAM3's include/ORBVocabulary.h, which named the forked
// DBoW2::TemplatedVocabulary directly.
// =============================================================================
#pragma once

#include <DBoW2/FORB.h>

#include "orbslam3r/dbow2_ext/text_vocabulary.hpp"

namespace orbslam3r
{

    using ORBVocabulary = dbow2_ext::TextFileVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

} // namespace orbslam3r
