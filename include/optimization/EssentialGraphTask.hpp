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

#ifndef ESSENTIALGRAPHTASK_H
#define ESSENTIALGRAPHTASK_H

#include "optim/Sim3GraphProblem.hpp"
#include "optim/Sim3GraphSolver.hpp"
#include "optimization/Digest.hpp"
#include "optimization/KeyFrameAndPose.hpp"

#include <sophus/se3.hpp>

#include <map>
#include <set>
#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    class MapPoint;

    typedef std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> Sim3Vector;

    // Optimizer::OptimizeEssentialGraph after a loop, in three steps. The graph
    // is every keyframe of the map, held to its neighbours by what each was to
    // the other before the loop was closed: its parent in the spanning tree,
    // the keyframes it closed loops with, those it shares a hundred points
    // with, and the keyframe before it in an inertial map. The new loop's
    // connections are what pulls it.
    class EssentialGraphTask
    {
    public:
        void Build(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, const KeyFrameAndPose &NonCorrectedSim3,
                   const KeyFrameAndPose &CorrectedSim3,
                   const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections, bool bFixScale);

        void Solve(optim::Sim3GraphSolver &solver);

        // Every keyframe takes its pose, scale divided out, and every point is
        // moved as its reference keyframe moved. Takes the map's update lock.
        void Apply(Map* pMap, KeyFrame* pCurKF) const;

        // For the build that runs v1.0's body beside this (Shadow.hpp): what
        // Build read, to be asked before Solve; what Apply would write, to be
        // asked before v1.0 writes; and whether the map then holds it.
        Digest Input() const;
        struct Written
        {
            std::vector<Sophus::SE3f> vTiw;   // by keyframe of the map as listed
            std::vector<Eigen::Vector3f> vXw; // by point likewise
            std::vector<unsigned char> vbPoint;
        };
        Written Preview(KeyFrame* pCurKF) const;
        bool Matches(const Written &written) const;

    private:
        void Write(KeyFrame* pCurKF, Written* pPreview) const;

        optim::Sim3GraphProblem mProblem;
        std::vector<KeyFrame*> mvpKFs; // the map's, as listed
        std::vector<MapPoint*> mvpMPs;
        std::vector<KeyFrame*> mvpPoseKF; // by pose of the problem
        std::vector<int> mvnPoseOfId;     // by keyframe id: its pose in the problem, or -1
        Sim3Vector mvScw;                 // by keyframe id: its pose before the optimisation
    };

    // Optimizer::OptimizeEssentialGraph after a merge. Three groups of
    // keyframes: those of the map merged into, held; those of the merged map
    // already moved into it by the welding, held; and the rest of the merged
    // map, which this moves. Two keyframes constrain each other when both have
    // a pose in the new frame or both have one in the old.
    class MergeGraphTask
    {
    public:
        void Build(KeyFrame* pCurKF, const std::vector<KeyFrame*> &vpFixedKFs,
                   const std::vector<KeyFrame*> &vpFixedCorrectedKFs, const std::vector<KeyFrame*> &vpNonFixedKFs);

        void Solve(optim::Sim3GraphSolver &solver);

        // The keyframes that were not held take their poses, keeping the one
        // they had as mTcwBefMerge; the points given are moved as their
        // reference keyframes were. Takes the map's update lock.
        void Apply(KeyFrame* pCurKF, const std::vector<KeyFrame*> &vpNonFixedKFs,
                   const std::vector<MapPoint*> &vpNonCorrectedMPs) const;

        // For the build that runs v1.0's body beside this: as above, of the
        // poses only -- the points are moved by what the keyframes then hold.
        Digest Input() const;
        std::vector<Sophus::SE3f> Preview(const std::vector<KeyFrame*> &vpNonFixedKFs) const;
        bool Matches(const std::vector<KeyFrame*> &vpNonFixedKFs, const std::vector<Sophus::SE3f> &vTiw) const;

    private:
        optim::Sim3GraphProblem mProblem;
        std::vector<KeyFrame*> mvpPoseKF;
        std::vector<int> mvnPoseOfId;
        std::vector<bool> mvbBadPose; // by keyframe id: it has a pose in the old frame
    };

} // namespace ORB_SLAM3

#endif // ESSENTIALGRAPHTASK_H
