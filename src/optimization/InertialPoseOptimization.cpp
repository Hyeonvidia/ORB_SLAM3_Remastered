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

#include "optimization/Optimizer.hpp"
#include "tracking/Frame.hpp"

#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <orbslam3r/g2o_ext/compat.hpp>
#include <orbslam3r/g2o_ext/solver_factory.hpp>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "optimization/G2oTypes.hpp"
#include "common/Converter.hpp"

#include <mutex>

#include "optimization/OptimizableTypes.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "optimization/KeyFrameAndPose.hpp"
#include "common/Verbose.hpp"

namespace ORB_SLAM3
{
    Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
    {
        // Goal
        // a  | ab | ac       a*  | 0 | ac*
        // ba | b  | bc  -->  0   | 0 | 0
        // ca | cb | c        ca* | 0 | c*

        // Size of block before block to marginalize
        const int a = start;
        // Size of block to marginalize
        const int b = end - start + 1;
        // Size of block after block to marginalize
        const int c = H.cols() - (end + 1);

        // Reorder as follows:
        // a  | ab | ac       a  | ac | ab
        // ba | b  | bc  -->  ca | c  | cb
        // ca | cb | c        ba | bc | b

        Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(), H.cols());
        if(a > 0)
        {
            Hn.block(0, 0, a, a) = H.block(0, 0, a, a);
            Hn.block(0, a + c, a, b) = H.block(0, a, a, b);
            Hn.block(a + c, 0, b, a) = H.block(a, 0, b, a);
        }
        if(a > 0 && c > 0)
        {
            Hn.block(0, a, a, c) = H.block(0, a + b, a, c);
            Hn.block(a, 0, c, a) = H.block(a + b, 0, c, a);
        }
        if(c > 0)
        {
            Hn.block(a, a, c, c) = H.block(a + b, a + b, c, c);
            Hn.block(a, a + c, c, b) = H.block(a + b, a, c, b);
            Hn.block(a + c, a, b, c) = H.block(a, a + b, b, c);
        }
        Hn.block(a + c, a + c, b, b) = H.block(a, a, b, b);

        // Perform marginalization (Schur complement)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a + c, a + c, b, b), Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues();
        for(int i = 0; i < b; ++i)
        {
            if(singularValues_inv(i) > 1e-6)
                singularValues_inv(i) = 1.0 / singularValues_inv(i);
            else
                singularValues_inv(i) = 0;
        }
        Eigen::MatrixXd invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
        Hn.block(0, 0, a + c, a + c) = Hn.block(0, 0, a + c, a + c) -
                                       Hn.block(0, a + c, a + c, b) * invHb * Hn.block(a + c, 0, b, a + c);
        Hn.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
        Hn.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
        Hn.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

        // Inverse reorder
        // a*  | ac* | 0       a*  | 0 | ac*
        // ca* | c*  | 0  -->  0   | 0 | 0
        // 0   | 0   | 0       ca* | 0 | c*
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(), H.cols());
        if(a > 0)
        {
            res.block(0, 0, a, a) = Hn.block(0, 0, a, a);
            res.block(0, a, a, b) = Hn.block(0, a + c, a, b);
            res.block(a, 0, b, a) = Hn.block(a + c, 0, b, a);
        }
        if(a > 0 && c > 0)
        {
            res.block(0, a + b, a, c) = Hn.block(0, a, a, c);
            res.block(a + b, 0, c, a) = Hn.block(a, 0, c, a);
        }
        if(c > 0)
        {
            res.block(a + b, a + b, c, c) = Hn.block(a, a, c, c);
            res.block(a + b, a, c, b) = Hn.block(a, a + c, c, b);
            res.block(a, a + b, b, c) = Hn.block(a + c, a, b, c);
        }

        res.block(a, a, b, b) = Hn.block(a + c, a + c, b, b);

        return res;
    }

    int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;

        auto* solver =
            orbslam3r::g2o_ext::MakeGaussNewton<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kDense>();
        optimizer.setVerbose(false);
        optimizer.setAlgorithm(solver);

        int nInitialMonoCorrespondences = 0;
        int nInitialStereoCorrespondences = 0;
        int nInitialCorrespondences = 0;

        // Set Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->N;
        const int Nleft = pFrame->Nleft;
        const bool bRight = (Nleft != -1);

        std::vector<EdgeMonoOnlyPose*> vpEdgesMono;
        std::vector<EdgeStereoOnlyPose*> vpEdgesStereo;
        std::vector<size_t> vnIndexEdgeMono;
        std::vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        {
            std::lock_guard<std::mutex> lock(MapPoint::mGlobalMutex);

            for(int i = 0; i < N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    cv::KeyPoint kpUn;

                    // Left monocular observation
                    if((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft)
                    {
                        if(i < Nleft) // pair left-right
                            kpUn = pFrame->mvKeys[i];
                        else
                            kpUn = pFrame->mvKeysUn[i];

                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 0);

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    // Stereo observation
                    else if(!bRight)
                    {
                        nInitialStereoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        kpUn = pFrame->mvKeysUn[i];
                        const float kp_ur = pFrame->mvuRight[i];
                        Eigen::Matrix<double, 3, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }

                    // Right monocular observation
                    if(bRight && i >= Nleft)
                    {
                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        kpUn = pFrame->mvKeysRight[i - Nleft];
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 1);

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                }
            }
        }
        nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        KeyFrame* pKF = pFrame->mpLastKeyFrame;
        VertexPose* VPk = new VertexPose(pKF);
        VPk->setId(4);
        VPk->setFixed(true);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pKF);
        VVk->setId(5);
        VVk->setFixed(true);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pKF);
        VGk->setId(6);
        VGk->setFixed(true);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pKF);
        VAk->setId(7);
        VAk->setFixed(true);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0, VGk);
        egr->setVertex(1, VG);
        Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
        egr->setInformation(InfoG);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0, VAk);
        ear->setVertex(1, VA);
        Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
        ear->setInformation(InfoA);
        optimizer.addEdge(ear);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        float chi2Mono[4] = {12, 7.5, 5.991, 5.991};
        float chi2Stereo[4] = {15.6, 9.8, 7.815, 7.815};

        int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        int nBadMono = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersStereo = 0;
        int nInliers = 0;
        for(size_t it = 0; it < 4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            nBadMono = 0;
            nBadStereo = 0;
            nInliers = 0;
            nInliersMono = 0;
            nInliersStereo = 0;
            float chi2close = 1.5 * chi2Mono[it];

            // For monocular observations
            for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

                if((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->isDepthPositive())
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                    nInliersMono++;
                }

                if(it == 2)
                    e->setRobustKernel(0);
            }

            // For stereo observations
            for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                EdgeStereoOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2 > chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1); // not included in next optimization
                    nBadStereo++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it == 2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo;
            nBad = nBadMono + nBadStereo;

            if(optimizer.edges().size() < 10)
            {
                break;
            }
        }

        // If not too much tracks, recover not too bad points
        if((nInliers < 30) && !bRecInit)
        {
            nBad = 0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyPose* e1;
            EdgeStereoOnlyPose* e2;
            for(size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if(e1->chi2() < chi2MonoOut)
                    pFrame->mvbOutlier[idx] = false;
                else
                    nBad++;
            }
            for(size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if(e2->chi2() < chi2StereoOut)
                    pFrame->mvbOutlier[idx] = false;
                else
                    nBad++;
            }
        }

        // Recover optimized pose, velocity and biases
        pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(),
                                   VV->estimate().cast<float>());
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

        // Recover Hessian, marginalize keyFframe states and generate new prior for frame
        Eigen::Matrix<double, 15, 15> H;
        H.setZero();

        H.block<9, 9>(0, 0) += ei->GetHessian2();
        H.block<3, 3>(9, 9) += egr->GetHessian2();
        H.block<3, 3>(12, 12) += ear->GetHessian2();

        int tot_in = 0, tot_out = 0;
        for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6, 6>(0, 0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6, 6>(0, 0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb, VP->estimate().twb, VV->estimate(), VG->estimate(),
                                              VA->estimate(), H);

        return nInitialCorrespondences - nBad;
    }

    int Optimizer::PoseInertialOptimizationLastFrame(Frame* pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;

        auto* solver =
            orbslam3r::g2o_ext::MakeGaussNewton<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kDense>();
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        int nInitialMonoCorrespondences = 0;
        int nInitialStereoCorrespondences = 0;
        int nInitialCorrespondences = 0;

        // Set Current Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->N;
        const int Nleft = pFrame->Nleft;
        const bool bRight = (Nleft != -1);

        std::vector<EdgeMonoOnlyPose*> vpEdgesMono;
        std::vector<EdgeStereoOnlyPose*> vpEdgesStereo;
        std::vector<size_t> vnIndexEdgeMono;
        std::vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = std::sqrt(5.991);
        const float thHuberStereo = std::sqrt(7.815);

        {
            std::lock_guard<std::mutex> lock(MapPoint::mGlobalMutex);

            for(int i = 0; i < N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    cv::KeyPoint kpUn;
                    // Left monocular observation
                    if((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft)
                    {
                        if(i < Nleft) // pair left-right
                            kpUn = pFrame->mvKeys[i];
                        else
                            kpUn = pFrame->mvKeysUn[i];

                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 0);

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    // Stereo observation
                    else if(!bRight)
                    {
                        nInitialStereoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        kpUn = pFrame->mvKeysUn[i];
                        const float kp_ur = pFrame->mvuRight[i];
                        Eigen::Matrix<double, 3, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }

                    // Right monocular observation
                    if(bRight && i >= Nleft)
                    {
                        nInitialMonoCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        kpUn = pFrame->mvKeysRight[i - Nleft];
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 1);

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                }
            }
        }

        nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        // Set Previous Frame Vertex
        Frame* pFp = pFrame->mpPrevFrame;

        VertexPose* VPk = new VertexPose(pFp);
        VPk->setId(4);
        VPk->setFixed(false);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pFp);
        VVk->setId(5);
        VVk->setFixed(false);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pFp);
        VGk->setId(6);
        VGk->setFixed(false);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pFp);
        VAk->setId(7);
        VAk->setFixed(false);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0, VGk);
        egr->setVertex(1, VG);
        Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
        egr->setInformation(InfoG);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0, VAk);
        ear->setVertex(1, VA);
        Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
        ear->setInformation(InfoA);
        optimizer.addEdge(ear);

        if(!pFp->mpcpi)
            Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + std::to_string(pFp->mnId),
                               Verbose::VERBOSITY_NORMAL);

        EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

        ep->setVertex(0, VPk);
        ep->setVertex(1, VVk);
        ep->setVertex(2, VGk);
        ep->setVertex(3, VAk);
        g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
        ep->setRobustKernel(rkp);
        rkp->setDelta(5);
        optimizer.addEdge(ep);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        int nBadMono = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersStereo = 0;
        int nInliers = 0;
        for(size_t it = 0; it < 4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            nBadMono = 0;
            nBadStereo = 0;
            nInliers = 0;
            nInliersMono = 0;
            nInliersStereo = 0;
            float chi2close = 1.5 * chi2Mono[it];

            for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];
                bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->isDepthPositive())
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                    nInliersMono++;
                }

                if(it == 2)
                    e->setRobustKernel(0);
            }

            for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                EdgeStereoOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2 > chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBadStereo++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it == 2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo;
            nBad = nBadMono + nBadStereo;

            if(optimizer.edges().size() < 10)
            {
                break;
            }
        }

        if((nInliers < 30) && !bRecInit)
        {
            nBad = 0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyPose* e1;
            EdgeStereoOnlyPose* e2;
            for(size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if(e1->chi2() < chi2MonoOut)
                    pFrame->mvbOutlier[idx] = false;
                else
                    nBad++;
            }
            for(size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if(e2->chi2() < chi2StereoOut)
                    pFrame->mvbOutlier[idx] = false;
                else
                    nBad++;
            }
        }

        nInliers = nInliersMono + nInliersStereo;

        // Recover optimized pose, velocity and biases
        pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(),
                                   VV->estimate().cast<float>());
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

        // Recover Hessian, marginalize previous frame states and generate new prior for frame
        Eigen::Matrix<double, 30, 30> H;
        H.setZero();

        H.block<24, 24>(0, 0) += ei->GetHessian();

        Eigen::Matrix<double, 6, 6> Hgr = egr->GetHessian();
        H.block<3, 3>(9, 9) += Hgr.block<3, 3>(0, 0);
        H.block<3, 3>(9, 24) += Hgr.block<3, 3>(0, 3);
        H.block<3, 3>(24, 9) += Hgr.block<3, 3>(3, 0);
        H.block<3, 3>(24, 24) += Hgr.block<3, 3>(3, 3);

        Eigen::Matrix<double, 6, 6> Har = ear->GetHessian();
        H.block<3, 3>(12, 12) += Har.block<3, 3>(0, 0);
        H.block<3, 3>(12, 27) += Har.block<3, 3>(0, 3);
        H.block<3, 3>(27, 12) += Har.block<3, 3>(3, 0);
        H.block<3, 3>(27, 27) += Har.block<3, 3>(3, 3);

        H.block<15, 15>(0, 0) += ep->GetHessian();

        int tot_in = 0, tot_out = 0;
        for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6, 6>(15, 15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->mvbOutlier[idx])
            {
                H.block<6, 6>(15, 15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        H = Marginalize(H, 0, 14);

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb, VP->estimate().twb, VV->estimate(), VG->estimate(),
                                              VA->estimate(), H.block<15, 15>(15, 15));
        delete pFp->mpcpi;
        pFp->mpcpi = NULL;

        return nInitialCorrespondences - nBad;
    }

} // namespace ORB_SLAM3
