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

#ifndef FEATUREGRID_H
#define FEATUREGRID_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ORB_SLAM3
{

    // Which features fall in each cell of the grid laid over an image to speed
    // up matching, flat: cell c = column * rows + row holds
    // indices[offsets[c] .. offsets[c + 1]), in the features' own order. As an
    // array of vectors, one to a cell, it was 103 KB a keyframe, most of it
    // vector headers and a small heap block per occupied cell, and every copy
    // of a Frame -- there are four to a frame tracked -- made them all again;
    // it is 20 KB in two blocks.
    struct FeatureGrid
    {
        std::vector<std::uint32_t> offsets;
        std::vector<std::uint32_t> indices;

        bool empty() const { return offsets.empty(); }

        const std::uint32_t* begin(std::size_t c) const { return indices.data() + offsets[c]; }
        const std::uint32_t* end(std::size_t c) const { return indices.data() + offsets[c + 1]; }

        // cells[i]: the cell of feature i, for i in [0, n), or -1 for none.
        void Assign(std::size_t nCells, const int* cells, std::size_t n)
        {
            offsets.assign(nCells + 1, 0);
            for(std::size_t i = 0; i < n; ++i)
                if(cells[i] >= 0)
                    ++offsets[cells[i] + 1];
            for(std::size_t c = 0; c < nCells; ++c)
                offsets[c + 1] += offsets[c];
            indices.resize(offsets[nCells]);
            std::vector<std::uint32_t> next(offsets.begin(), offsets.end() - 1);
            for(std::size_t i = 0; i < n; ++i)
                if(cells[i] >= 0)
                    indices[next[cells[i]]++] = static_cast<std::uint32_t>(i);
        }
    };

} // namespace ORB_SLAM3

#endif // FEATUREGRID_H
