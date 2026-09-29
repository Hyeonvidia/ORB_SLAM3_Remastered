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

#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include <orbslam3r/dbow2_ext/compact_vocabulary.hpp>

namespace ORB_SLAM3
{

    // The vocabulary tree in arrays, 59 MB of them, where DBoW2's
    // TemplatedVocabulary holds a node object for each of its 1.1 million
    // nodes, 106 MB; the same words and the same scores. See
    // vendor_ext/dbow2_ext's compact_vocabulary.hpp. DBoW2's own, which can
    // also create a vocabulary and save it, is orbslam3r::ORBVocabulary.
    typedef orbslam3r::dbow2_ext::CompactVocabulary ORBVocabulary;

} // namespace ORB_SLAM3

#endif // ORBVOCABULARY_H
