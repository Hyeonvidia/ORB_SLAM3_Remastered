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

#ifndef FLATMAP_H
#define FLATMAP_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace ORB_SLAM3
{

    // A map for a handful of entries that is copied more often than it is
    // changed: the entries in one block, in the order of their keys. It is
    // read as a std::map is -- find, count, [], erase, and iterators over
    // pairs in the same order -- so that what used one can use this.
    //
    // What it is for: a map point's observations. A std::map of them is a
    // heap block of 64 bytes for each, and every reader takes a copy under
    // the point's mutex, block by block; this is 16 bytes each and a copy is
    // one block. Inserting and erasing move what follows, which for a
    // handful is nothing, and invalidate iterators, which a std::map's do not.
    template<class K, class V>
    class FlatMap
    {
    public:
        typedef K key_type;
        typedef V mapped_type;
        typedef std::pair<K, V> value_type;
        typedef typename std::vector<value_type>::iterator iterator;
        typedef typename std::vector<value_type>::const_iterator const_iterator;

        iterator begin() { return mvEntries.begin(); }
        iterator end() { return mvEntries.end(); }
        const_iterator begin() const { return mvEntries.begin(); }
        const_iterator end() const { return mvEntries.end(); }

        std::size_t size() const { return mvEntries.size(); }
        std::size_t capacity() const { return mvEntries.capacity(); }
        bool empty() const { return mvEntries.empty(); }
        void clear() { mvEntries.clear(); }

        iterator find(const K &key)
        {
            const iterator it = Lower(key);
            return (it != mvEntries.end() && it->first == key) ? it : mvEntries.end();
        }

        const_iterator find(const K &key) const
        {
            const const_iterator it = std::lower_bound(mvEntries.begin(), mvEntries.end(), key, Before);
            return (it != mvEntries.end() && it->first == key) ? it : mvEntries.end();
        }

        std::size_t count(const K &key) const { return find(key) != mvEntries.end() ? 1 : 0; }

        V &operator[](const K &key)
        {
            iterator it = Lower(key);
            if(it == mvEntries.end() || !(it->first == key))
                it = mvEntries.insert(it, value_type(key, V()));
            return it->second;
        }

        std::size_t erase(const K &key)
        {
            const iterator it = find(key);
            if(it == mvEntries.end())
                return 0;
            mvEntries.erase(it);
            return 1;
        }

        iterator erase(const_iterator it) { return mvEntries.erase(it); }

    private:
        static bool Before(const value_type &entry, const K &key) { return std::less<K>()(entry.first, key); }

        iterator Lower(const K &key) { return std::lower_bound(mvEntries.begin(), mvEntries.end(), key, Before); }

        std::vector<value_type> mvEntries;
    };

} // namespace ORB_SLAM3

#endif // FLATMAP_H
