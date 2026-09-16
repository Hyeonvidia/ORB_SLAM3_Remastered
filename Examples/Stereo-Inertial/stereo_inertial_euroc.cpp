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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>


#include"System.hpp"
#include "ImuTypes.hpp"
#include "Optimizer.hpp"

#include <string>
#include <vector>


void LoadImages(const std::string &strPathLeft, const std::string &strPathRight, const std::string &strPathTimes,
                std::vector<std::string> &vstrImageLeft, std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps);

void LoadIMU(const std::string &strImuPath, std::vector<double> &vTimeStamps, std::vector<cv::Point3f> &vAcc, std::vector<cv::Point3f> &vGyro);


int main(int argc, char **argv)
{
    if(argc < 5)
    {
        std::cerr << std::endl << "Usage: ./stereo_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) " << std::endl;
        return 1;
    }

    const int num_seq = (argc-3)/2;
    std::cout << "num_seq = " << num_seq << std::endl;
    bool bFileName= (((argc-3) % 2) == 1);
    std::string file_name;
    if (bFileName)
    {
        file_name = std::string(argv[argc-1]);
        std::cout << "file name: " << file_name << std::endl;
    }

    // Load all sequences:
    int seq;
    std::vector< std::vector<std::string> > vstrImageLeft;
    std::vector< std::vector<std::string> > vstrImageRight;
    std::vector< std::vector<double> > vTimestampsCam;
    std::vector< std::vector<cv::Point3f> > vAcc, vGyro;
    std::vector< std::vector<double> > vTimestampsImu;
    std::vector<int> nImages;
    std::vector<int> nImu;
    std::vector<int> first_imu(num_seq,0);

    vstrImageLeft.resize(num_seq);
    vstrImageRight.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        std::cout << "Loading images for sequence " << seq << "...";

        std::string pathSeq(argv[(2*seq) + 3]);
        std::string pathTimeStamps(argv[(2*seq) + 4]);

        std::string pathCam0 = pathSeq + "/mav0/cam0/data";
        std::string pathCam1 = pathSeq + "/mav0/cam1/data";
        std::string pathImu = pathSeq + "/mav0/imu0/data.csv";

        LoadImages(pathCam0, pathCam1, pathTimeStamps, vstrImageLeft[seq], vstrImageRight[seq], vTimestampsCam[seq]);
        std::cout << "LOADED!" << std::endl;

        std::cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        std::cout << "LOADED!" << std::endl;

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            std::cerr << "ERROR: Failed to load images or IMU for sequence" << seq << std::endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return -1;
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    std::cout << std::endl << "-------" << std::endl;
    std::cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, false);

    cv::Mat imLeft, imRight;
    for (seq = 0; seq<num_seq; seq++)
    {
        // Seq loop
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        double t_rect = 0.f;
        double t_resize = 0.f;
        double t_track = 0.f;
        int num_rect = 0;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read left and right images from file
            imLeft = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_UNCHANGED);
            imRight = cv::imread(vstrImageRight[seq][ni],cv::IMREAD_UNCHANGED);

            if(imLeft.empty())
            {
                std::cerr << std::endl << "Failed to load image at: "
                     << std::string(vstrImageLeft[seq][ni]) << std::endl;
                return 1;
            }

            if(imRight.empty())
            {
                std::cerr << std::endl << "Failed to load image at: "
                     << std::string(vstrImageRight[seq][ni]) << std::endl;
                return 1;
            }

            double tframe = vTimestampsCam[seq][ni];

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0)
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni]) // while(vTimestampsImu[first_imu]<=vTimestampsCam[ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            // Pass the images to the SLAM system
            SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
            t_track = t_rect + t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }

        if(seq < num_seq - 1)
        {
            std::cout << "Changing the dataset" << std::endl;

            SLAM.ChangeDataset();
        }


    }
    // Stop all threads
    SLAM.Shutdown();


    // Save camera trajectory
    if (bFileName)
    {
        const std::string kf_file =  "kf_" + std::string(argv[argc-1]) + ".txt";
        const std::string f_file =  "f_" + std::string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

void LoadImages(const std::string &strPathLeft, const std::string &strPathRight, const std::string &strPathTimes,
                std::vector<std::string> &vstrImageLeft, std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps)
{
    std::ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        std::string s;
        std::getline(fTimes,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}

void LoadIMU(const std::string &strImuPath, std::vector<double> &vTimeStamps, std::vector<cv::Point3f> &vAcc, std::vector<cv::Point3f> &vGyro)
{
    std::ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while(!fImu.eof())
    {
        std::string s;
        std::getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            std::string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                item = s.substr(0, pos);
                data[count++] = std::stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = std::stod(item);

            vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
}
