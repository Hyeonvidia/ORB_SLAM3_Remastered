#include "dbow2_wrapper.hpp"

namespace slam3 { namespace dbow2 {

bool ORBVocabulary::loadFromTextFile(const std::string &filename)
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

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(descriptors.rows);
    for(int j = 0; j < descriptors.rows; j++)
        vDesc.push_back(descriptors.row(j));
    return vDesc;
}

int descriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for(int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

}} // namespace slam3::dbow2
