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

#ifndef BOWWORDS_H
#define BOWWORDS_H

#include <cmath>

#include <DBoW2/BowVector.h>

#include "atlas/ORBVocabulary.hpp"
#include "common/FlatMap.hpp"

namespace ORB_SLAM3
{

    // The words of an image and their weights, as a Frame and a KeyFrame keep
    // them. DBoW2's BowVector is a std::map: for the 2,000 features of a KITTI
    // image, 2,000 heap blocks of 64 bytes, 126 KB -- more than a third of what
    // a keyframe held. The same entries in one block are 32.
    typedef FlatMap<DBoW2::WordId, DBoW2::WordValue> BowWords;

    // What voc.score() says of the same words: DBoW2's L1 score, term by term
    // in the same order, or DBoW2 itself for a vocabulary that scores otherwise.
    inline double Score(const ORBVocabulary &voc, const BowWords &v1, const BowWords &v2)
    {
        if(voc.getScoringType() != DBoW2::L1_NORM)
        {
            DBoW2::BowVector b1, b2;
            for(const BowWords::value_type &word : v1)
                b1.insert(b1.end(), word);
            for(const BowWords::value_type &word : v2)
                b2.insert(b2.end(), word);
            return voc.score(b1, b2);
        }

        BowWords::const_iterator it1 = v1.begin(), it2 = v2.begin();
        const BowWords::const_iterator end1 = v1.end(), end2 = v2.end();
        double score = 0;
        while(it1 != end1 && it2 != end2)
        {
            if(it1->first == it2->first)
            {
                score += std::fabs(it1->second - it2->second) - std::fabs(it1->second) - std::fabs(it2->second);
                ++it1;
                ++it2;
            }
            else if(it1->first < it2->first)
                it1 = v1.lower_bound(it2->first);
            else
                it2 = v2.lower_bound(it1->first);
        }
        return -score / 2.0;
    }

} // namespace ORB_SLAM3

#endif // BOWWORDS_H
