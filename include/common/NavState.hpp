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

#ifndef NAVSTATE_H
#define NAVSTATE_H

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "common/ImuTypes.hpp"

namespace ORB_SLAM3
{

    // The pose-and-velocity state that a Frame and a KeyFrame each carry.
    //
    // Both classes held the same thing and derived the same quantities from it, but
    // not in the same way: Frame cached the camera centre and recomputed the IMU
    // pose on every read, KeyFrame cached the IMU centre but only when a
    // calibration was set -- leaving it uninitialised otherwise -- and each of them
    // refreshed its own derived matrices by hand. Putting the state in one type
    // makes SetPose the single place they are refreshed, so they cannot drift.
    //
    // Everything past Tcw is derived from it. Nothing here is independently
    // settable, which is the point: an inconsistent Rcw was reachable before.
    //
    // Deliberately carries no mutex. A Frame is a transient value read by the
    // thread that made it; a KeyFrame is shared by four threads and guards this
    // behind mMutexPose. The locking policy belongs to the owner, not to the state
    // -- which is why this is composed into both rather than inherited by one from
    // the other.
    class NavState
    {
    public:
        // Sets the camera pose and refreshes every form derived from it.
        // Taken by value so that SetPose(state.Tcw()) is safe; the argument would
        // otherwise alias the member being overwritten.
        void SetPose(Sophus::SE3f Tcw, const IMU::Calib &calib);

        // Sets the IMU pose, which fixes the camera pose through the extrinsics.
        void SetImuPose(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const IMU::Calib &calib);

        void SetVelocity(const Eigen::Vector3f &Vw);

        const Sophus::SE3f &Tcw() const { return mTcw; }
        const Sophus::SE3f &Twc() const { return mTwc; }
        const Eigen::Matrix3f &Rcw() const { return mRcw; }
        const Eigen::Matrix3f &Rwc() const { return mRwc; }
        Eigen::Vector3f tcw() const { return mTcw.translation(); }

        // Camera centre in world coordinates.
        Eigen::Vector3f Ow() const { return mTwc.translation(); }

        // IMU body in world coordinates. Cached, because the inertial optimisers
        // ask for it per keyframe per iteration.
        const Eigen::Vector3f &Owb() const { return mOwb; }

        Sophus::SE3f ImuPose(const IMU::Calib &calib) const { return mTwc * calib.mTcb; }
        Eigen::Matrix3f ImuRotation(const IMU::Calib &calib) const { return (mTwc * calib.mTcb).rotationMatrix(); }

        const Eigen::Vector3f &Velocity() const { return mVw; }

        bool HasPose() const { return mbHasPose; }
        bool HasVelocity() const { return mbHasVelocity; }

    private:
        // KeyFrame's archive layout is fixed by the maps already on disk, so its
        // serialize() reads and writes these members in place rather than going
        // through the accessors. PostLoad then calls SetPose to rebuild the rest.
        friend class KeyFrame;

        Sophus::SE3f mTcw;
        Sophus::SE3f mTwc;
        Eigen::Matrix3f mRcw = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f mRwc = Eigen::Matrix3f::Identity();
        Eigen::Vector3f mOwb = Eigen::Vector3f::Zero();

        // IMU linear velocity, world frame.
        Eigen::Vector3f mVw = Eigen::Vector3f::Zero();

        bool mbHasPose = false;
        bool mbHasVelocity = false;
    };

} // namespace ORB_SLAM3

#endif // NAVSTATE_H
