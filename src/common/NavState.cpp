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

#include "common/NavState.hpp"

namespace ORB_SLAM3
{

    void NavState::SetPose(Sophus::SE3f Tcw, const IMU::Calib &calib)
    {
        mTcw = Tcw;
        mTwc = mTcw.inverse();
        mRcw = mTcw.rotationMatrix();
        mRwc = mTwc.rotationMatrix();

        // Unconditionally, unlike KeyFrame::SetPose before this, which skipped the
        // update when no calibration was set and left mOwb holding whatever the
        // allocation happened to contain. With no calibration mTcb is the identity,
        // so this is the camera centre -- a sane value rather than an uninitialised
        // read for anything that asks.
        mOwb = mRwc * calib.mTcb.translation() + mTwc.translation();

        mbHasPose = true;
    }

    void NavState::SetImuPose(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const IMU::Calib &calib)
    {
        const Sophus::SE3f Twb(Rwb, twb);
        SetPose(calib.mTcb * Twb.inverse(), calib);
    }

    void NavState::SetVelocity(const Eigen::Vector3f &Vw)
    {
        mVw = Vw;
        mbHasVelocity = true;
    }

} // namespace ORB_SLAM3
