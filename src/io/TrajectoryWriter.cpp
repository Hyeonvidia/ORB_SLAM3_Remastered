/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
* Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
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

#include "TrajectoryWriter.hpp"
#include "Converter.hpp"
#include "LocalMapping.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace ORB_SLAM3
{

void TrajectoryWriter::SaveTrajectoryTUM(const std::string &filename)
{
    std::cout << std::endl << "Saving camera trajectory to " << filename << " ..." << std::endl;
    if(mSensor==MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << std::endl;
        return;
    }

    std::vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(std::list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();

        Eigen::Vector3f twc = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();

        f << std::setprecision(6) << *lT << " " <<  std::setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
}

void TrajectoryWriter::SaveKeyFrameTrajectoryTUM(const std::string &filename)
{
    std::cout << std::endl << "Saving keyframe trajectory to " << filename << " ..." << std::endl;

    std::vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        Eigen::Vector3f t = Twc.translation();
        f << std::setprecision(6) << pKF->mTimeStamp << std::setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

    }

    f.close();
}

void TrajectoryWriter::SaveTrajectoryEuRoC(const std::string &filename)
{

    std::cout << std::endl << "Saving trajectory to " << filename << " ..." << std::endl;
    /*if(mSensor==MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << std::endl;
        return;
    }*/

    std::vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    int numMaxKFs = 0;
    Map* pBiggerMap;
    std::cout << "There are " << std::to_string(vpMaps.size()) << " maps in the atlas" << std::endl;
    for(Map* pMap :vpMaps)
    {
        std::cout << "  Map " << std::to_string(pMap->GetId()) << " has " << std::to_string(pMap->GetAllKeyFrames().size()) << " KFs" << std::endl;
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    std::vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b depending on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    // std::cout << "file open" << std::endl;
    f << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //std::cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << std::endl;
    //std::cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << std::endl;
    //std::cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << std::endl;
    //std::cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << std::endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //std::cout << "1" << std::endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //std::cout << "KF: " << pKF->mnId << std::endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //std::cout << "2.5" << std::endl;

        while(pKF->isBad())
        {
            //std::cout << " 2.bad" << std::endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //std::cout << "--Parent KF: " << pKF->mnId << std::endl;
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //std::cout << "--Parent KF is from another map" << std::endl;
            continue;
        }

        //std::cout << "3" << std::endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // std::cout << "4" << std::endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << std::setprecision(6) << 1e9*(*lT) << " " <<  std::setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << std::setprecision(6) << 1e9*(*lT) << " " <<  std::setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }

        // std::cout << "5" << std::endl;
    }
    //std::cout << "end saving trajectory" << std::endl;
    f.close();
    std::cout << std::endl << "End of saving trajectory to " << filename << " ..." << std::endl;
}

void TrajectoryWriter::SaveTrajectoryEuRoC(const std::string &filename, Map* pMap)
{

    std::cout << std::endl << "Saving trajectory of map " << pMap->GetId() << " to " << filename << " ..." << std::endl;
    /*if(mSensor==MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << std::endl;
        return;
    }*/

    int numMaxKFs = 0;

    std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    // std::cout << "file open" << std::endl;
    f << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //std::cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << std::endl;
    //std::cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << std::endl;
    //std::cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << std::endl;
    //std::cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << std::endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //std::cout << "1" << std::endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //std::cout << "KF: " << pKF->mnId << std::endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //std::cout << "2.5" << std::endl;

        while(pKF->isBad())
        {
            //std::cout << " 2.bad" << std::endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //std::cout << "--Parent KF: " << pKF->mnId << std::endl;
        }

        if(!pKF || pKF->GetMap() != pMap)
        {
            //std::cout << "--Parent KF is from another map" << std::endl;
            continue;
        }

        //std::cout << "3" << std::endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // std::cout << "4" << std::endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << std::setprecision(6) << 1e9*(*lT) << " " <<  std::setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << std::setprecision(6) << 1e9*(*lT) << " " <<  std::setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }

        // std::cout << "5" << std::endl;
    }
    //std::cout << "end saving trajectory" << std::endl;
    f.close();
    std::cout << std::endl << "End of saving trajectory to " << filename << " ..." << std::endl;
}

void TrajectoryWriter::SaveKeyFrameTrajectoryEuRoC(const std::string &filename)
{
    std::cout << std::endl << "Saving keyframe trajectory to " << filename << " ..." << std::endl;

    std::vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap && pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    if(!pBiggerMap)
    {
        std::cout << "There is not a map!!" << std::endl;
        return;
    }

    std::vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << std::setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  std::setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << std::setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    f.close();
}

void TrajectoryWriter::SaveKeyFrameTrajectoryEuRoC(const std::string &filename, Map* pMap)
{
    std::cout << std::endl << "Saving keyframe trajectory of map " << pMap->GetId() << " to " << filename << " ..." << std::endl;

    std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << std::setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  std::setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << std::setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    f.close();
}

void TrajectoryWriter::SaveTrajectoryKITTI(const std::string &filename)
{
    std::cout << std::endl << "Saving camera trajectory to " << filename << " ..." << std::endl;
    if(mSensor==MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << std::endl;
        return;
    }

    std::vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Tow = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(std::list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        if(!pKF)
            continue;

        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Tow;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        Eigen::Vector3f twc = Twc.translation();

        f << std::setprecision(9) << Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
             Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
             Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << std::endl;
    }
    f.close();
}


void TrajectoryWriter::SaveDebugData(const int &initIdx)
{
    // 0. Save initialization trajectory
    SaveTrajectoryEuRoC("init_FrameTrajectoy_" +std::to_string(mpLocalMapper->mInitSect)+ "_" + std::to_string(initIdx)+".txt");

    // 1. Save scale
    std::ofstream f;
    f.open("init_Scale_" + std::to_string(mpLocalMapper->mInitSect) + ".txt", std::ios_base::app);
    f << std::fixed;
    f << mpLocalMapper->mScale << std::endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" +std::to_string(mpLocalMapper->mInitSect)+ ".txt", std::ios_base::app);
    f << std::fixed;
    f << mpLocalMapper->mRwg(0,0) << "," << mpLocalMapper->mRwg(0,1) << "," << mpLocalMapper->mRwg(0,2) << std::endl;
    f << mpLocalMapper->mRwg(1,0) << "," << mpLocalMapper->mRwg(1,1) << "," << mpLocalMapper->mRwg(1,2) << std::endl;
    f << mpLocalMapper->mRwg(2,0) << "," << mpLocalMapper->mRwg(2,1) << "," << mpLocalMapper->mRwg(2,2) << std::endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" +std::to_string(mpLocalMapper->mInitSect)+ ".txt", std::ios_base::app);
    f << std::fixed;
    f << mpLocalMapper->mCostTime << std::endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" +std::to_string(mpLocalMapper->mInitSect)+ ".txt", std::ios_base::app);
    f << std::fixed;
    f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << std::endl;
    f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << std::endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" +std::to_string(mpLocalMapper->mInitSect)+ "_" +std::to_string(initIdx)+".txt", std::ios_base::app);
    f << std::fixed;
    for(int i=0; i<mpLocalMapper->mcovInertial.rows(); i++)
    {
        for(int j=0; j<mpLocalMapper->mcovInertial.cols(); j++)
        {
            if(j!=0)
                f << ",";
            f << std::setprecision(15) << mpLocalMapper->mcovInertial(i,j);
        }
        f << std::endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" +std::to_string(mpLocalMapper->mInitSect)+ ".txt", std::ios_base::app);
    f << std::fixed;
    f << mpLocalMapper->mInitTime << std::endl;
    f.close();
}

} //namespace ORB_SLAM3
