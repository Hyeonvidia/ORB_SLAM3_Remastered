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

#ifndef INERTIALBATASK_H
#define INERTIALBATASK_H

#include "common/ImuTypes.hpp"
#include "optim/InertialBaProblem.hpp"
#include "optim/InertialBundleAdjuster.hpp"
#include "optimization/Digest.hpp"
#include "optimization/KeyFrameAndPose.hpp"

#include <deque>
#include <map>
#include <utility>
#include <vector>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    class MapPoint;

    // What the three inertial adjustments share: the problem, the copies of
    // the preintegrations it points at, and which keyframe and point each
    // state and point of it is.
    struct InertialBaWindow
    {
        optim::InertialBaProblem problem;
        std::deque<IMU::Preintegrated> preintegrations;

        std::map<long unsigned int, int> stateOfId; // by keyframe id
        std::vector<KeyFrame*> stateKF;             // by state of the problem
        std::map<MapPoint*, int> pointOf;
        std::vector<MapPoint*> pointMP; // by point of the problem

        // By observation.
        std::vector<KeyFrame*> obsKF;
        std::vector<MapPoint*> obsMP;
    };

    // Optimizer::LocalInertialBA of a keyframe in three steps. The window is
    // the keyframe and those before it in time -- ten, or twenty-five when
    // bLarge -- each tied to the one before by what the inertial unit
    // measured; the points they see; the keyframe before the first of them,
    // held with its velocity and biases; and up to two hundred other keyframes
    // that see those points, held.
    class LocalInertialBaTask
    {
    public:
        void Build(KeyFrame* pKF, bool bLarge, bool bRecInit);

        // Ten iterations, or four when bLarge. Nothing ends them early.
        void Solve(optim::InertialBundleAdjuster &solver);

        // Nothing, if the cost more than doubled or is not a number and the
        // window is not a large one. Otherwise an observation that does not
        // fit is erased -- chi2 over 5.991 for two numbers, half as much again
        // for a point nearer than ten metres, 7.815 for three, or the point
        // behind the camera -- and the states and the points are written.
        // Takes the map's update lock.
        void Apply(Map* pMap);

        // For the build that runs v1.0's body beside this (Shadow.hpp): the
        // observations Apply would erase, and what was read of the points to
        // tell; whether Apply would do nothing; whether the map holds what it
        // would write; and the marks Build left taken off again.
        const std::vector<std::pair<KeyFrame*, MapPoint*>> &Classify();
        Digest Judged() const { return mJudged; }
        bool Failed() const;
        bool Matches(const std::vector<std::pair<KeyFrame*, MapPoint*>> &erased, bool bFailed) const;
        void ResetMarks();

    private:
        InertialBaWindow mWindow;
        bool mbLarge = false;
        int mnIterations = 10;

        std::vector<KeyFrame*> mvpLocalKF; // in the order v1.0 walked them
        std::vector<KeyFrame*> mvpFixedKF;
        std::vector<MapPoint*> mvpPoints;
        std::vector<KeyFrame*> mvpMarkedLocal;
        std::vector<KeyFrame*> mvpMarkedFixed;

        std::vector<std::pair<KeyFrame*, MapPoint*>> mvToErase;
        Digest mJudged;
    };

    // Optimizer::FullInertialBA over every keyframe and point of a map, in
    // three steps: after a loop, and when the inertial unit is tied to the
    // map. With bInit the keyframes share one pair of biases, held towards
    // zero by priorG and priorA; otherwise each has its own, tied to those of
    // the keyframe before by how far a bias may walk.
    class FullInertialBaTask
    {
    public:
        // False, and nothing to solve, when bFixLocal leaves fewer than three
        // keyframes free.
        bool Build(Map* pMap, bool bFixLocal, bool bInit, float priorG, float priorA);

        void Solve(optim::InertialBundleAdjuster &solver, int nIterations, bool* pbStopFlag);

        // To the keyframes and points themselves when nLoopId is 0; otherwise
        // beside them (mTcwGBA, mVwbGBA, mBiasGBA, mPosGBA), for whoever asked
        // to apply.
        void Apply(unsigned long nLoopId) const;

        // For the build that runs v1.0's body beside this: what Build read of
        // the map, to be asked before Solve, and whether the map holds what
        // Apply would write.
        Digest Input() const;
        bool Matches(unsigned long nLoopId) const;

    private:
        InertialBaWindow mWindow;
        Map* mpMap = nullptr;
        bool mbInit = false;

        std::vector<KeyFrame*> mvpKF; // the map's, as listed
        std::vector<int> mvnState;    // the state of each in the problem, or -1
    };

    // Optimizer::MergeInertialBA in three steps: the adjustment that welds two
    // inertial maps. The window is the current keyframe and those before it,
    // the keyframe it was matched to with those around it in time, and up to
    // thirty keyframes that see the same points; one keyframe of the old map
    // is held.
    class MergeInertialBaTask
    {
    public:
        void Build(KeyFrame* pCurrKF, KeyFrame* pMergeKF);

        // Eight iterations at most; pbStopFlag ends them early.
        void Solve(optim::InertialBundleAdjuster &solver, bool* pbStopFlag);

        // An observation whose chi2 is over 5.991 for two numbers or 7.815 for
        // three is erased; the states and points are written, and each
        // keyframe's pose is put in corrPoses. Takes the map's update lock.
        void Apply(Map* pMap, KeyFrameAndPose &corrPoses);

        // For the build that runs v1.0's body beside this: as above.
        const std::vector<std::pair<KeyFrame*, MapPoint*>> &Classify();
        bool Matches(const std::vector<std::pair<KeyFrame*, MapPoint*>> &erased) const;
        void ResetMarks();

    private:
        InertialBaWindow mWindow;

        std::vector<KeyFrame*> mvpLocalKF; // in the order v1.0 walked them
        std::vector<KeyFrame*> mvpCovKF;
        std::vector<MapPoint*> mvpPoints;
        std::vector<KeyFrame*> mvpMarkedLocal;
        std::vector<KeyFrame*> mvpMarkedFixed;

        std::vector<std::pair<KeyFrame*, MapPoint*>> mvToErase;
    };

} // namespace ORB_SLAM3

#endif // INERTIALBATASK_H
