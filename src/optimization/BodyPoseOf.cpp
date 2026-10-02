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

#include "optimization/BodyPoseOf.hpp"

#include "atlas/KeyFrame.hpp"
#include "tracking/Frame.hpp"

namespace ORB_SLAM3
{

    optim::BodyPose BodyPoseOf(KeyFrame* pKF)
    {
        optim::BodyPose pose;

        // Load IMU pose
        pose.twb = pKF->GetImuPosition().cast<double>();
        pose.Rwb = pKF->GetImuRotation().cast<double>();

        // Load camera poses
        pose.nCameras = pKF->mpCamera2 ? 2 : 1;

        // Left camera
        pose.tcw[0] = pKF->GetTranslation().cast<double>();
        pose.Rcw[0] = pKF->GetRotation().cast<double>();
        pose.tcb[0] = pKF->mImuCalib.mTcb.translation().cast<double>();
        pose.Rcb[0] = pKF->mImuCalib.mTcb.rotationMatrix().cast<double>();
        pose.Rbc[0] = pose.Rcb[0].transpose();
        pose.tbc[0] = pKF->mImuCalib.mTbc.translation().cast<double>();
        pose.camera[0] = pKF->mpCamera;
        pose.bf = pKF->mbf;

        if(pose.nCameras > 1)
        {
            Eigen::Matrix4d Trl = pKF->GetRelativePoseTrl().matrix().cast<double>();
            pose.Rcw[1] = Trl.block<3, 3>(0, 0) * pose.Rcw[0];
            pose.tcw[1] = Trl.block<3, 3>(0, 0) * pose.tcw[0] + Trl.block<3, 1>(0, 3);
            pose.tcb[1] = Trl.block<3, 3>(0, 0) * pose.tcb[0] + Trl.block<3, 1>(0, 3);
            pose.Rcb[1] = Trl.block<3, 3>(0, 0) * pose.Rcb[0];
            pose.Rbc[1] = pose.Rcb[1].transpose();
            pose.tbc[1] = -pose.Rbc[1] * pose.tcb[1];
            pose.camera[1] = pKF->mpCamera2;
        }
        return pose;
    }

    optim::BodyPose BodyPoseOf(Frame* pF)
    {
        optim::BodyPose pose;

        // Load IMU pose
        pose.twb = pF->GetImuPosition().cast<double>();
        pose.Rwb = pF->GetImuRotation().cast<double>();

        // Load camera poses
        pose.nCameras = pF->mpCamera2 ? 2 : 1;

        // Left camera
        pose.tcw[0] = pF->GetPose().translation().cast<double>();
        pose.Rcw[0] = pF->GetPose().rotationMatrix().cast<double>();
        pose.tcb[0] = pF->mImuCalib.mTcb.translation().cast<double>();
        pose.Rcb[0] = pF->mImuCalib.mTcb.rotationMatrix().cast<double>();
        pose.Rbc[0] = pose.Rcb[0].transpose();
        pose.tbc[0] = pF->mImuCalib.mTbc.translation().cast<double>();
        pose.camera[0] = pF->mpCamera;
        pose.bf = pF->mbf;

        if(pose.nCameras > 1)
        {
            Eigen::Matrix4d Trl = pF->GetRelativePoseTrl().matrix().cast<double>();
            pose.Rcw[1] = Trl.block<3, 3>(0, 0) * pose.Rcw[0];
            pose.tcw[1] = Trl.block<3, 3>(0, 0) * pose.tcw[0] + Trl.block<3, 1>(0, 3);
            pose.tcb[1] = Trl.block<3, 3>(0, 0) * pose.tcb[0] + Trl.block<3, 1>(0, 3);
            pose.Rcb[1] = Trl.block<3, 3>(0, 0) * pose.Rcb[0];
            pose.Rbc[1] = pose.Rcb[1].transpose();
            pose.tbc[1] = -pose.Rbc[1] * pose.tcb[1];
            pose.camera[1] = pF->mpCamera2;
        }
        return pose;
    }

    optim::BodyPose BodyPoseOf(const Eigen::Matrix3d &_Rwc, const Eigen::Vector3d &_twc, KeyFrame* pKF)
    {
        optim::BodyPose pose;

        // This is only for posegrpah, we do not care about multicamera
        pose.nCameras = 1;

        pose.tcb[0] = pKF->mImuCalib.mTcb.translation().cast<double>();
        pose.Rcb[0] = pKF->mImuCalib.mTcb.rotationMatrix().cast<double>();
        pose.Rbc[0] = pose.Rcb[0].transpose();
        pose.tbc[0] = pKF->mImuCalib.mTbc.translation().cast<double>();
        pose.twb = _Rwc * pose.tcb[0] + _twc;
        pose.Rwb = _Rwc * pose.Rcb[0];
        pose.Rcw[0] = _Rwc.transpose();
        pose.tcw[0] = -pose.Rcw[0] * _twc;
        pose.camera[0] = pKF->mpCamera;
        pose.bf = pKF->mbf;
        return pose;
    }

} // namespace ORB_SLAM3
