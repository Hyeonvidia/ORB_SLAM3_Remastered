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

#include "optim/InertialBundleAdjuster.hpp"

#include "optim_g2o/InertialTypes.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <orbslam3r/g2o_ext/compat.hpp>
#include <orbslam3r/g2o_ext/solver_factory.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace ORB_SLAM3
{

    namespace optim
    {

        namespace
        {

            // The graph v1.0 built in Optimizer::LocalInertialBA, FullInertialBA
            // and MergeInertialBA. Its vertex ids put the poses first, then
            // velocity, gyroscope bias and accelerometer bias keyframe by
            // keyframe, then the shared biases, then the points, each group in
            // the order of the keyframes' or points' ids; these ids put them
            // in the same order with the problem's indices. The edges are
            // added in the order of the problem's terms.
            class G2oInertialBundleAdjuster : public InertialBundleAdjuster
            {
            public:
                void Solve(InertialBaProblem &problem, const SolveOptions &options) override
                {
                    g2o::SparseOptimizer optimizer;

                    auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX,
                                                                     orbslam3r::g2o_ext::LinearSolver::kEigen>();
                    if(options.damping == SolveOptions::kValue)
                        solver->setUserLambdaInit(options.dampingValue);
                    else if(options.damping == SolveOptions::kFractionOfDiagonal)
                        solver->setInitialDamping(options.dampingValue);
                    if(options.nStallIterations > 0)
                        solver->setStallIterations(options.nStallIterations);
                    optimizer.setAlgorithm(solver);
                    optimizer.setVerbose(false);

                    if(options.pbStop)
                        optimizer.setForceStopFlag(options.pbStop);

                    // Set KeyFrame vertices
                    const int n = static_cast<int>(problem.states.size());
                    std::vector<VertexPose*> vpPose(n);
                    std::vector<VertexVelocity*> vpVelocity(n, nullptr);
                    std::vector<VertexGyroBias*> vpGyroBias(n, nullptr);
                    std::vector<VertexAccBias*> vpAccBias(n, nullptr);
                    for(int i = 0; i < n; i++)
                    {
                        const InertialState &state = problem.states[i];
                        const bool bFixed = problem.fixed[i];

                        VertexPose* VP = new VertexPose(state.pose);
                        VP->setId(i);
                        VP->setFixed(bFixed);
                        optimizer.addVertex(VP);
                        vpPose[i] = VP;

                        if(problem.inertial[i])
                        {
                            VertexVelocity* VV = new VertexVelocity(state.velocity);
                            VV->setId(n + 3 * i);
                            VV->setFixed(bFixed);
                            optimizer.addVertex(VV);
                            vpVelocity[i] = VV;
                            if(!problem.sharedBias)
                            {
                                VertexGyroBias* VG = new VertexGyroBias(state.gyroBias);
                                VG->setId(n + 3 * i + 1);
                                VG->setFixed(bFixed);
                                optimizer.addVertex(VG);
                                vpGyroBias[i] = VG;
                                VertexAccBias* VA = new VertexAccBias(state.accBias);
                                VA->setId(n + 3 * i + 2);
                                VA->setFixed(bFixed);
                                optimizer.addVertex(VA);
                                vpAccBias[i] = VA;
                            }
                        }
                    }

                    VertexGyroBias* VGshared = nullptr;
                    VertexAccBias* VAshared = nullptr;
                    if(problem.sharedBias)
                    {
                        VGshared = new VertexGyroBias(problem.sharedGyroBias);
                        VGshared->setId(4 * n);
                        VGshared->setFixed(false);
                        optimizer.addVertex(VGshared);
                        VAshared = new VertexAccBias(problem.sharedAccBias);
                        VAshared->setId(4 * n + 1);
                        VAshared->setFixed(false);
                        optimizer.addVertex(VAshared);
                    }

                    // IMU links
                    for(std::size_t k = 0; k < problem.terms(); k++)
                    {
                        const int i = problem.from[k];
                        const int j = problem.to[k];

                        EdgeInertial* ei = new EdgeInertial(problem.preintegration[k]);
                        ei->setVertex(0, vpPose[i]);
                        ei->setVertex(1, vpVelocity[i]);
                        ei->setVertex(2, problem.sharedBias ? VGshared : vpGyroBias[i]);
                        ei->setVertex(3, problem.sharedBias ? VAshared : vpAccBias[i]);
                        ei->setVertex(4, vpPose[j]);
                        ei->setVertex(5, vpVelocity[j]);

                        if(problem.inertialHuber[k] > 0.0)
                        {
                            g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                            ei->setRobustKernel(rki);
                            rki->setDelta(problem.inertialHuber[k]);
                        }
                        if(problem.inertialScale[k] != 1.0)
                            ei->setInformation(ei->information() * problem.inertialScale[k]);

                        optimizer.addEdge(ei);

                        if(!problem.sharedBias)
                        {
                            EdgeGyroRW* egr = new EdgeGyroRW();
                            egr->setVertex(0, vpGyroBias[i]);
                            egr->setVertex(1, vpGyroBias[j]);
                            egr->setInformation(problem.gyroWalkInformation[k]);
                            optimizer.addEdge(egr);

                            EdgeAccRW* ear = new EdgeAccRW();
                            ear->setVertex(0, vpAccBias[i]);
                            ear->setVertex(1, vpAccBias[j]);
                            ear->setInformation(problem.accWalkInformation[k]);
                            optimizer.addEdge(ear);
                        }
                    }

                    if(problem.sharedBias && problem.biasPriors)
                    {
                        // Add prior to comon biases
                        Eigen::Vector3f bprior;
                        bprior.setZero();

                        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
                        epa->setVertex(0, VAshared);
                        epa->setInformation(problem.accPriorInformation * Eigen::Matrix3d::Identity());
                        optimizer.addEdge(epa);

                        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
                        epg->setVertex(0, VGshared);
                        epg->setInformation(problem.gyroPriorInformation * Eigen::Matrix3d::Identity());
                        optimizer.addEdge(epg);
                    }

                    // Set MapPoint vertices
                    const int nPoints = static_cast<int>(problem.Xw.size());
                    std::vector<g2o::VertexSBAPointXYZ*> vpPoint(nPoints);
                    for(int j = 0; j < nPoints; j++)
                    {
                        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
                        vPoint->setEstimate(problem.Xw[j]);
                        vPoint->setId(4 * n + 2 + j);
                        vPoint->setMarginalized(true);
                        optimizer.addVertex(vPoint);
                        vpPoint[j] = vPoint;
                    }

                    // Create visual constraints
                    const std::size_t nObs = problem.observations();
                    std::vector<EdgeMono*> vpMono(nObs, nullptr);
                    std::vector<EdgeStereo*> vpStereo(nObs, nullptr);
                    for(std::size_t k = 0; k < nObs; k++)
                    {
                        g2o::OptimizableGraph::Edge* pEdge = nullptr;
                        if(problem.kind[k] == kStereo)
                        {
                            EdgeStereo* e = new EdgeStereo(0);

                            e->setVertex(0, vpPoint[problem.point[k]]);
                            e->setVertex(1, vpPose[problem.state[k]]);
                            e->setMeasurement(problem.uv[k]);
                            e->setInformation(Eigen::Matrix3d::Identity() * problem.invSigma2[k]);
                            vpStereo[k] = e;
                            pEdge = e;
                        }
                        else
                        {
                            EdgeMono* e = new EdgeMono(problem.kind[k] == kRight ? 1 : 0);

                            e->setVertex(0, vpPoint[problem.point[k]]);
                            e->setVertex(1, vpPose[problem.state[k]]);
                            e->setMeasurement(problem.uv[k].head<2>());
                            e->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2[k]);
                            vpMono[k] = e;
                            pEdge = e;
                        }

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        pEdge->setRobustKernel(rk);
                        rk->setDelta(problem.huber[k]);

                        optimizer.addEdge(pEdge);
                    }

                    optimizer.initializeOptimization();
                    optimizer.computeActiveErrors();
                    problem.costBefore = optimizer.activeRobustChi2();
                    optimizer.optimize(options.nIterations);
                    problem.costAfter = optimizer.activeRobustChi2();

                    for(std::size_t k = 0; k < nObs; k++)
                    {
                        if(vpMono[k])
                        {
                            problem.chi2[k] = vpMono[k]->chi2();
                            problem.depthPositive[k] = vpMono[k]->isDepthPositive();
                        }
                        else
                        {
                            problem.chi2[k] = vpStereo[k]->chi2();
                            problem.depthPositive[k] = 1;
                        }
                    }

                    // Recover optimized data
                    for(int i = 0; i < n; i++)
                    {
                        InertialState &state = problem.states[i];
                        const ImuCamPose &found = vpPose[i]->estimate();
                        state.pose.Rwb = found.Rwb;
                        state.pose.twb = found.twb;
                        for(int c = 0; c < state.pose.nCameras; c++)
                        {
                            state.pose.Rcw[c] = found.Rcw[c];
                            state.pose.tcw[c] = found.tcw[c];
                        }
                        if(vpVelocity[i])
                            state.velocity = vpVelocity[i]->estimate();
                        if(vpGyroBias[i])
                            state.gyroBias = vpGyroBias[i]->estimate();
                        if(vpAccBias[i])
                            state.accBias = vpAccBias[i]->estimate();
                    }
                    if(problem.sharedBias)
                    {
                        problem.sharedGyroBias = VGshared->estimate();
                        problem.sharedAccBias = VAshared->estimate();
                    }
                    for(int j = 0; j < nPoints; j++)
                        problem.Xw[j] = vpPoint[j]->estimate();
                }
            };

        } // namespace

        std::unique_ptr<InertialBundleAdjuster> MakeInertialBundleAdjuster()
        {
            return std::make_unique<G2oInertialBundleAdjuster>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
