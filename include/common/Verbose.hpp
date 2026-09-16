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

#pragma once

#include <iostream>
#include <string>

namespace ORB_SLAM3
{

/// Level-filtered logging.
///
/// This used to live in System.hpp, which meant that every file wanting to
/// print a line had to depend on the whole System class -- and most of them got
/// it by accident, through some other header's include. It is its own header
/// now so that dependency says what it means.
class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET = 0,
        VERBOSITY_NORMAL = 1,
        VERBOSITY_VERBOSE = 2,
        VERBOSITY_VERY_VERBOSE = 3,
        VERBOSITY_DEBUG = 4
    };

    static eLevel th;

    static void PrintMess(const std::string &str, eLevel lev)
    {
        if(lev <= th)
        {
            std::cout << str << std::endl;
        }
    }

    static void SetTh(eLevel _th) { th = _th; }
};

} // namespace ORB_SLAM3
