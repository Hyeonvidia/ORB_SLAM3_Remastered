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

#ifndef MEMORYAUDIT_H
#define MEMORYAUDIT_H

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

    class Atlas;
    class KeyFrame;
    class MapPoint;

    // Where the memory is, by structure rather than by call stack: what every
    // KeyFrame and MapPoint holds, member by member, for the ones still in a map
    // and for the ones that were culled -- ORB-SLAM3 erases those from the map
    // and never deletes them. Off unless ORBSLAM3R_MEMORY_REPORT=1, and then
    // printed once by System::Shutdown(). When it is off, the cost is one read of
    // a static bool in each constructor and destructor.
    //
    // Sizes are what the allocator hands out, not what was asked for: glibc
    // rounds a request up to 16 bytes after adding 8 of its own, which matters
    // when most allocations are a grid cell holding one index.
    class MemoryAudit
    {
    public:
        typedef std::map<std::string, std::size_t> Footprint;

        static bool Enabled();

        static void Register(const KeyFrame* pKF);
        static void Unregister(const KeyFrame* pKF);
        static void Register(const MapPoint* pMP);
        static void Unregister(const MapPoint* pMP);

        // vocabulary: what ORBVocabulary::MemoryFootprint() returned.
        static void Report(std::ostream &os, const Footprint &vocabulary);

        // A heap block of n bytes as glibc malloc accounts for it.
        static std::size_t Chunk(std::size_t n)
        {
            return n == 0 ? 0 : ((n + 8 + 15) / 16 * 16 < 32 ? 32 : (n + 8 + 15) / 16 * 16);
        }

        template<class T>
        static std::size_t Vector(const std::vector<T> &v)
        {
            return Chunk(v.capacity() * sizeof(T));
        }

        // A std::map or std::set: one node per element, 32 bytes of tree links
        // and then the value.
        template<class M>
        static std::size_t Tree(const M &m)
        {
            return m.size() * Chunk(32 + sizeof(typename M::value_type));
        }

        // cv::Mat data; the header is part of whatever object holds the Mat.
        static std::size_t Mat(const cv::Mat &m) { return m.data && m.u ? Chunk(m.total() * m.elemSize() + 64) : 0; }
    };

} // namespace ORB_SLAM3

#endif // MEMORYAUDIT_H
