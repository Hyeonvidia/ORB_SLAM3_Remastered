#include "FeatureExtractor.hpp"
#include "ORBextractor.hpp"

#include <list>
#include <algorithm>

namespace ORB_SLAM3 {

// =============================================================================
// OctTreeDistributor — Original ORB-SLAM3 quadtree distribution
// =============================================================================
std::vector<cv::KeyPoint> OctTreeDistributor::distribute(
    const std::vector<cv::KeyPoint>& vToDistributeKeys,
    int minX, int maxX, int minY, int maxY,
    int nFeatures, int /*level*/)
{
    const int nIni = std::round(static_cast<float>(maxX - minX) / (maxY - minY));
    const float hX = static_cast<float>(maxX - minX) / nIni;

    std::list<ExtractorNode> lNodes;
    std::vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++) {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    auto lit = lNodes.begin();
    while (lit != lNodes.end()) {
        if (lit->vKeys.size() == 1) {
            lit->bNoMore = true;
            lit++;
        } else if (lit->vKeys.empty()) {
            lit = lNodes.erase(lit);
        } else {
            lit++;
        }
    }

    bool bFinish = false;
    int iteration = 0;
    std::vector<std::pair<int, ExtractorNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish) {
        iteration++;
        int prevSize = lNodes.size();
        lit = lNodes.begin();
        int nToExpand = 0;
        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end()) {
            if (lit->bNoMore) {
                lit++;
                continue;
            }

            ExtractorNode n1, n2, n3, n4;
            lit->DivideNode(n1, n2, n3, n4);

            if (n1.vKeys.size() > 0) {
                lNodes.push_front(n1);
                if (n1.vKeys.size() > 1) {
                    nToExpand++;
                    vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                    lNodes.front().lit = lNodes.begin();
                }
            }
            if (n2.vKeys.size() > 0) {
                lNodes.push_front(n2);
                if (n2.vKeys.size() > 1) {
                    nToExpand++;
                    vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                    lNodes.front().lit = lNodes.begin();
                }
            }
            if (n3.vKeys.size() > 0) {
                lNodes.push_front(n3);
                if (n3.vKeys.size() > 1) {
                    nToExpand++;
                    vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                    lNodes.front().lit = lNodes.begin();
                }
            }
            if (n4.vKeys.size() > 0) {
                lNodes.push_front(n4);
                if (n4.vKeys.size() > 1) {
                    nToExpand++;
                    vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                    lNodes.front().lit = lNodes.begin();
                }
            }

            lit = lNodes.erase(lit);
            continue;
        }

        if ((int)lNodes.size() >= nFeatures || (int)lNodes.size() == prevSize) {
            bFinish = true;
        } else if (((int)lNodes.size() + nToExpand * 3) > nFeatures) {
            while (!bFinish) {
                prevSize = lNodes.size();
                std::vector<std::pair<int, ExtractorNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                std::sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
                    ExtractorNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    if (n1.vKeys.size() > 0) {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0) {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0) {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0) {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= nFeatures)
                        break;
                }

                if ((int)lNodes.size() >= nFeatures || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nFeatures);
    for (auto& node : lNodes) {
        auto& vNodeKeys = node.vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;
        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }
        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

// =============================================================================
// GridDistributor — Simple grid-based alternative
// =============================================================================
std::vector<cv::KeyPoint> GridDistributor::distribute(
    const std::vector<cv::KeyPoint>& vToDistributeKeys,
    int minX, int maxX, int minY, int maxY,
    int nFeatures, int /*level*/)
{
    if (vToDistributeKeys.empty() || nFeatures <= 0)
        return {};

    int gridCols = std::max(1, static_cast<int>(std::sqrt(nFeatures)));
    int gridRows = std::max(1, nFeatures / gridCols);
    float cellW = static_cast<float>(maxX - minX) / gridCols;
    float cellH = static_cast<float>(maxY - minY) / gridRows;

    // For each cell, keep the keypoint with highest response
    std::vector<cv::KeyPoint> result;
    result.reserve(nFeatures);

    std::vector<std::vector<cv::KeyPoint>> grid(gridCols * gridRows);
    for (const auto& kp : vToDistributeKeys) {
        int col = std::min(static_cast<int>(kp.pt.x / cellW), gridCols - 1);
        int row = std::min(static_cast<int>(kp.pt.y / cellH), gridRows - 1);
        grid[row * gridCols + col].push_back(kp);
    }

    for (auto& cell : grid) {
        if (cell.empty()) continue;
        auto best = std::max_element(cell.begin(), cell.end(),
            [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.response < b.response;
            });
        result.push_back(*best);
    }

    return result;
}

} // namespace ORB_SLAM3
