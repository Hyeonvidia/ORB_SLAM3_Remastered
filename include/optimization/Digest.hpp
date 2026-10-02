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

#ifndef OPTIMIZATION_DIGEST_H
#define OPTIMIZATION_DIGEST_H

#include <cstddef>
#include <cstdint>

namespace ORB_SLAM3
{

    // Whether two walks over the map read the same things: a hash of each
    // thing read, summed, so that the order they were read in does not matter.
    // What must be in order carries its place among the values hashed.
    class Digest
    {
    public:
        template<class... T>
        void Add(char tag, const T &... values)
        {
            std::uint64_t h = 1469598103934665603ull;
            Mix(h, tag);
            (Mix(h, values), ...);
            mSum += h;
        }

        bool operator==(const Digest &other) const { return mSum == other.mSum; }
        bool operator!=(const Digest &other) const { return mSum != other.mSum; }

    private:
        template<class T>
        static void Mix(std::uint64_t &h, const T &value)
        {
            const unsigned char* p = reinterpret_cast<const unsigned char*>(&value);
            for(std::size_t i = 0; i < sizeof(T); i++)
            {
                h ^= p[i];
                h *= 1099511628211ull;
            }
        }

        std::uint64_t mSum = 0;
    };

} // namespace ORB_SLAM3

#endif // OPTIMIZATION_DIGEST_H
