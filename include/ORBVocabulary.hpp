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

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

namespace ORB_SLAM3
{

// Wrapper class that extends TemplatedVocabulary with text file loading
// (upstream DBoW2 only supports YAML via cv::FileStorage)
class ORBVocabulary : public DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
{
public:
    using Base = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;
    using Base::Base;

    bool loadFromTextFile(const std::string &filename)
    {
        std::ifstream f;
        f.open(filename.c_str());

        if(f.eof())
            return false;

        m_words.clear();
        m_nodes.clear();

        std::string s;
        getline(f, s);
        std::stringstream ss;
        ss << s;
        ss >> m_k;
        ss >> m_L;
        int n1, n2;
        ss >> n1;
        ss >> n2;

        if(m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3)
        {
            std::cerr << "Vocabulary loading failure: This is not a correct text file!" << std::endl;
            return false;
        }

        m_scoring = (DBoW2::ScoringType)n1;
        m_weighting = (DBoW2::WeightingType)n2;
        createScoringObject();

        int expected_nodes =
            (int)((pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));
        m_nodes.reserve(expected_nodes);
        m_words.reserve(pow((double)m_k, (double)m_L + 1));

        m_nodes.resize(1);
        m_nodes[0].id = 0;

        while(!f.eof())
        {
            std::string snode;
            getline(f, snode);
            if(snode.empty()) break;

            std::stringstream ssnode;
            ssnode << snode;

            int nid = m_nodes.size();
            m_nodes.resize(m_nodes.size() + 1);
            m_nodes[nid].id = nid;

            int pid;
            ssnode >> pid;
            m_nodes[nid].parent = pid;
            m_nodes[pid].children.push_back(nid);

            int nIsLeaf;
            ssnode >> nIsLeaf;

            std::stringstream ssd;
            for(int iD = 0; iD < DBoW2::FORB::L; iD++)
            {
                std::string sElement;
                ssnode >> sElement;
                ssd << sElement << " ";
            }
            DBoW2::FORB::fromString(m_nodes[nid].descriptor, ssd.str());

            ssnode >> m_nodes[nid].weight;

            if(nIsLeaf > 0)
            {
                int wid = m_words.size();
                m_words.resize(wid + 1);

                m_nodes[nid].word_id = wid;
                m_words[wid] = &m_nodes[nid];
            }
            else
            {
                m_nodes[nid].word_id = 0;
            }
        }

        return true;
    }
};

} //namespace ORB_SLAM3

#endif // ORBVOCABULARY_H
