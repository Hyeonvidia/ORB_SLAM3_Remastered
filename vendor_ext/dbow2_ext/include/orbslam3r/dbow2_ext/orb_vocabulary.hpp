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

#include "orbslam3r/dbow2_ext/forb32.hpp"
#include "orbslam3r/dbow2_ext/text_vocabulary.hpp"

namespace orbslam3r
{

    // FORB32, not DBoW2::FORB: the same vocabulary in a third of the memory; see forb32.hpp.
    using ORBVocabulary = dbow2_ext::TextFileVocabulary<dbow2_ext::FORB32::TDescriptor, dbow2_ext::FORB32>;

} // namespace orbslam3r
