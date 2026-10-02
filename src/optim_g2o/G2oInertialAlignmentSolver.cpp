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

#include "optim/InertialAlignmentSolver.hpp"

#include "optim_g2o/InertialTypes.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
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

            // The graph v1.0 built in the three Optimizer::InertialOptimization:
            // a held pose and a velocity per keyframe, one gyroscope and one
            // accelerometer bias, the direction of gravity and the scale; the
            // two priors on the biases, then an EdgeInertialGS per inertial
            // term. The ids are in v1.0's order -- poses, velocities, biases,
            // gravity, scale -- and so are the edges. Sparse.
            class G2oInertialAlignmentSolver : public InertialAlignmentSolver
            {
            public:
                void Solve(InertialAlignmentProblem &problem, const SolveOptions &options) override
                {
                    g2o::SparseOptimizer optimizer;

                    if(options.algorithm == SolveOptions::kGaussNewton)
                    {
                        auto* solver = orbslam3r::g2o_ext::MakeGaussNewton<g2o::BlockSolverX,
                                                                           orbslam3r::g2o_ext::LinearSolver::kEigen>();
                        optimizer.setAlgorithm(solver);
                    }
                    else
                    {
                        auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX,
                                                                         orbslam3r::g2o_ext::LinearSolver::kEigen>();
                        if(options.damping == SolveOptions::kValue)
                            solver->setUserLambdaInit(options.dampingValue);
                        else if(options.damping == SolveOptions::kFractionOfDiagonal)
                            solver->setInitialDamping(options.dampingValue);
                        if(options.nStallIterations > 0)
                            solver->setStallIterations(options.nStallIterations);
                        optimizer.setAlgorithm(solver);
                    }
                    if(options.pbStop)
                        optimizer.setForceStopFlag(options.pbStop);

                    // Set KeyFrame vertices (fixed poses and optimizable velocities)
                    const int n = static_cast<int>(problem.poses.size());
                    std::vector<VertexPose*> vpPose(n);
                    std::vector<VertexVelocity*> vpVelocity(n);
                    for(int i = 0; i < n; i++)
                    {
                        VertexPose* VP = new VertexPose(problem.poses[i]);
                        VP->setId(i);
                        VP->setFixed(true);
                        optimizer.addVertex(VP);
                        vpPose[i] = VP;

                        VertexVelocity* VV = new VertexVelocity(problem.velocity[i]);
                        VV->setId(n + i);
                        VV->setFixed(problem.velocitiesFixed);
                        optimizer.addVertex(VV);
                        vpVelocity[i] = VV;
                    }

                    // Biases
                    VertexGyroBias* VG = new VertexGyroBias(problem.gyroBias);
                    VG->setId(2 * n);
                    VG->setFixed(problem.biasesFixed);
                    optimizer.addVertex(VG);
                    VertexAccBias* VA = new VertexAccBias(problem.accBias);
                    VA->setId(2 * n + 1);
                    VA->setFixed(problem.biasesFixed);
                    optimizer.addVertex(VA);

                    if(problem.biasPriors)
                    {
                        // prior acc bias
                        Eigen::Vector3f bprior;
                        bprior.setZero();

                        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
                        epa->setVertex(0, VA);
                        epa->setInformation(problem.accPriorInformation * Eigen::Matrix3d::Identity());
                        optimizer.addEdge(epa);
                        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
                        epg->setVertex(0, VG);
                        epg->setInformation(problem.gyroPriorInformation * Eigen::Matrix3d::Identity());
                        optimizer.addEdge(epg);
                    }

                    // Gravity and scale
                    VertexGDir* VGDir = new VertexGDir(problem.Rwg);
                    VGDir->setId(2 * n + 2);
                    VGDir->setFixed(problem.gravityFixed);
                    optimizer.addVertex(VGDir);
                    VertexScale* VS = new VertexScale(problem.scale);
                    VS->setId(2 * n + 3);
                    VS->setFixed(problem.scaleFixed);
                    optimizer.addVertex(VS);

                    // Graph edges
                    // IMU links with gravity and scale
                    for(std::size_t k = 0; k < problem.constraints(); k++)
                    {
                        const int i = problem.from[k];
                        const int j = problem.to[k];

                        EdgeInertialGS* ei = new EdgeInertialGS(problem.preintegration[k]);
                        ei->setVertex(0, vpPose[i]);
                        ei->setVertex(1, vpVelocity[i]);
                        ei->setVertex(2, VG);
                        ei->setVertex(3, VA);
                        ei->setVertex(4, vpPose[j]);
                        ei->setVertex(5, vpVelocity[j]);
                        ei->setVertex(6, VGDir);
                        ei->setVertex(7, VS);
                        if(problem.huber > 0.0)
                        {
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            ei->setRobustKernel(rk);
                            rk->setDelta(problem.huber);
                        }
                        optimizer.addEdge(ei);
                    }

                    optimizer.setVerbose(false);
                    optimizer.initializeOptimization();
                    optimizer.optimize(options.nIterations);

                    // Recover optimized data
                    problem.scale = VS->estimate();
                    problem.gyroBias = VG->estimate();
                    problem.accBias = VA->estimate();
                    problem.Rwg = VGDir->estimate().Rwg;
                    for(int i = 0; i < n; i++)
                        problem.velocity[i] = vpVelocity[i]->estimate();
                }
            };

        } // namespace

        std::unique_ptr<InertialAlignmentSolver> MakeInertialAlignmentSolver()
        {
            return std::make_unique<G2oInertialAlignmentSolver>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
