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

#include "optim_g2o/OptimizableTypes.hpp"

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
    void Optimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg,
                                         Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd &covInertial, bool bFixedVel,
                                         bool bGauss, float priorG, float priorA)
    {
        Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
        int its = 200;
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;

        auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kEigen>();

        if(priorG != 0.f)
            solver->setUserLambdaInit(1e3);

        optimizer.setAlgorithm(solver);

        // Set KeyFrame vertices (fixed poses and optimizable velocities)
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;
            VertexPose* VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + (pKFi->mnId) + 1);
            if(bFixedVel)
                VV->setFixed(true);
            else
                VV->setFixed(false);

            optimizer.addVertex(VV);
        }

        // Biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(maxKFid * 2 + 2);
        if(bFixedVel)
            VG->setFixed(true);
        else
            VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(maxKFid * 2 + 3);
        if(bFixedVel)
            VA->setFixed(true);
        else
            VA->setFixed(false);

        optimizer.addVertex(VA);
        // prior acc bias
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);
        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Rwg);
        VGDir->setId(maxKFid * 2 + 4);
        VGDir->setFixed(false);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(scale);
        VS->setId(maxKFid * 2 + 5);
        VS->setFixed(!bMono); // Fixed for stereo case
        optimizer.addVertex(VS);

        // Graph edges
        // IMU links with gravity and scale
        std::vector<EdgeInertialGS*> vpei;
        vpei.reserve(vpKFs.size());
        std::vector<std::pair<KeyFrame*, KeyFrame*>> vppUsedKF;
        vppUsedKF.reserve(vpKFs.size());
        //std::cout << "build optimization graph" << std::endl;

        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId <= maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                    continue;
                if(!pKFi->mpImuPreintegrated)
                    std::cout << "Not preintegrated measurement" << std::endl;

                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
                g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid * 2 + 2);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid * 2 + 3);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid * 2 + 4);
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid * 2 + 5);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    std::cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2
                              << ", " << VGDir << ", " << VS << std::endl;

                    continue;
                }
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                vpei.push_back(ei);

                vppUsedKF.push_back(std::make_pair(pKFi->mPrevKF, pKFi));
                optimizer.addEdge(ei);
            }
        }

        // Compute error for different scales
        std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(its);

        scale = VS->estimate();

        // Recover optimized data
        // Biases
        VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid * 2 + 2));
        VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid * 2 + 3));
        Vector6d vb;
        vb << VG->estimate(), VA->estimate();
        bg << VG->estimate();
        ba << VA->estimate();
        scale = VS->estimate();

        IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
        Rwg = VGDir->estimate().Rwg;

        //Keyframes velocities and biases
        const int N = vpKFs.size();
        for(size_t i = 0; i < N; i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;

            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
            Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
            pKFi->SetVelocity(Vw.cast<float>());

            if((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
            {
                pKFi->SetNewBias(b);
                if(pKFi->mpImuPreintegrated)
                    pKFi->mpImuPreintegrated->Reintegrate();
            }
            else
                pKFi->SetNewBias(b);
        }
    }

    void Optimizer::InertialOptimization(Map* pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG,
                                         float priorA)
    {
        int its = 200; // Check number of iterations
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;

        auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kEigen>();
        solver->setUserLambdaInit(1e3);

        optimizer.setAlgorithm(solver);

        // Set KeyFrame vertices (fixed poses and optimizable velocities)
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;
            VertexPose* VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + (pKFi->mnId) + 1);
            VV->setFixed(false);

            optimizer.addVertex(VV);
        }

        // Biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(maxKFid * 2 + 2);
        VG->setFixed(false);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(maxKFid * 2 + 3);
        VA->setFixed(false);

        optimizer.addVertex(VA);
        // prior acc bias
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);
        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
        VGDir->setId(maxKFid * 2 + 4);
        VGDir->setFixed(true);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(1.0);
        VS->setId(maxKFid * 2 + 5);
        VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
        optimizer.addVertex(VS);

        // Graph edges
        // IMU links with gravity and scale
        std::vector<EdgeInertialGS*> vpei;
        vpei.reserve(vpKFs.size());
        std::vector<std::pair<KeyFrame*, KeyFrame*>> vppUsedKF;
        vppUsedKF.reserve(vpKFs.size());

        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId <= maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                    continue;

                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
                g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid * 2 + 2);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid * 2 + 3);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid * 2 + 4);
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid * 2 + 5);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    std::cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2
                              << ", " << VGDir << ", " << VS << std::endl;

                    continue;
                }
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                vpei.push_back(ei);

                vppUsedKF.push_back(std::make_pair(pKFi->mPrevKF, pKFi));
                optimizer.addEdge(ei);
            }
        }

        // Compute error for different scales
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(its);

        // Recover optimized data
        // Biases
        VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid * 2 + 2));
        VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid * 2 + 3));
        Vector6d vb;
        vb << VG->estimate(), VA->estimate();
        bg << VG->estimate();
        ba << VA->estimate();

        IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

        //Keyframes velocities and biases
        const int N = vpKFs.size();
        for(size_t i = 0; i < N; i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;

            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
            Eigen::Vector3d Vw = VV->estimate();
            pKFi->SetVelocity(Vw.cast<float>());

            if((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
            {
                pKFi->SetNewBias(b);
                if(pKFi->mpImuPreintegrated)
                    pKFi->mpImuPreintegrated->Reintegrate();
            }
            else
                pKFi->SetNewBias(b);
        }
    }

    void Optimizer::InertialOptimization(Map* pMap, Eigen::Matrix3d &Rwg, double &scale)
    {
        int its = 10;
        long unsigned int maxKFid = pMap->GetMaxKFid();
        const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;

        auto* solver =
            orbslam3r::g2o_ext::MakeGaussNewton<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kEigen>();
        optimizer.setAlgorithm(solver);

        // Set KeyFrame vertices (all variables are fixed)
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(pKFi->mnId > maxKFid)
                continue;
            VertexPose* VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 1 + (pKFi->mnId));
            VV->setFixed(true);
            optimizer.addVertex(VV);

            // Vertex of fixed biases
            VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
            VG->setId(2 * (maxKFid + 1) + (pKFi->mnId));
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(vpKFs.front());
            VA->setId(3 * (maxKFid + 1) + (pKFi->mnId));
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }

        // Gravity and scale
        VertexGDir* VGDir = new VertexGDir(Rwg);
        VGDir->setId(4 * (maxKFid + 1));
        VGDir->setFixed(false);
        optimizer.addVertex(VGDir);
        VertexScale* VS = new VertexScale(scale);
        VS->setId(4 * (maxKFid + 1) + 1);
        VS->setFixed(false);
        optimizer.addVertex(VS);

        // Graph edges
        int count_edges = 0;
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            if(pKFi->mPrevKF && pKFi->mnId <= maxKFid)
            {
                if(pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                    continue;

                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid + 1) + pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid + 1) + pKFi->mnId);
                g2o::HyperGraph::Vertex* VG = optimizer.vertex(2 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VA = optimizer.vertex(3 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4 * (maxKFid + 1));
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(4 * (maxKFid + 1) + 1);
                if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
                {
                    Verbose::PrintMess("Error" + std::to_string(VP1->id()) + ", " + std::to_string(VV1->id()) + ", " +
                                           std::to_string(VG->id()) + ", " + std::to_string(VA->id()) + ", " +
                                           std::to_string(VP2->id()) + ", " + std::to_string(VV2->id()) + ", " +
                                           std::to_string(VGDir->id()) + ", " + std::to_string(VS->id()),
                                       Verbose::VERBOSITY_NORMAL);

                    continue;
                }
                count_edges++;
                EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
                ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
                ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rk);
                rk->setDelta(1.f);
                optimizer.addEdge(ei);
            }
        }

        // Compute error for different scales
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        float err = optimizer.activeRobustChi2();
        optimizer.optimize(its);
        optimizer.computeActiveErrors();
        float err_end = optimizer.activeRobustChi2();
        // Recover optimized data
        scale = VS->estimate();
        Rwg = VGDir->estimate().Rwg;
    }

} // namespace ORB_SLAM3
