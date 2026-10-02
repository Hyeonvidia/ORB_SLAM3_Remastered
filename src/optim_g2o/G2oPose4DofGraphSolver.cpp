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

#include "optim/Pose4DofGraphSolver.hpp"

#include "optim_g2o/InertialTypes.hpp"

#include <g2o/core/block_solver.h>
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

            // The graph v1.0 built in Optimizer::OptimizeEssentialGraph4DoF: a
            // VertexPose4DoF per pose and an Edge4DoF per constraint, on sparse
            // Levenberg-Marquardt. The ids are the problem's indices and the
            // edges are added in the order of the constraints.
            class G2oPose4DofGraphSolver : public Pose4DofGraphSolver
            {
            public:
                void Solve(Pose4DofGraphProblem &problem, const SolveOptions &options) override
                {
                    g2o::SparseOptimizer optimizer;
                    optimizer.setVerbose(false);
                    auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX,
                                                                     orbslam3r::g2o_ext::LinearSolver::kEigen>();
                    if(options.damping == SolveOptions::kValue)
                        solver->setUserLambdaInit(options.dampingValue);
                    else if(options.damping == SolveOptions::kFractionOfDiagonal)
                        solver->setInitialDamping(options.dampingValue);
                    if(options.nStallIterations > 0)
                        solver->setStallIterations(options.nStallIterations);
                    optimizer.setAlgorithm(solver);
                    if(options.pbStop)
                        optimizer.setForceStopFlag(options.pbStop);

                    const std::size_t nPoses = problem.poses.size();
                    std::vector<VertexPose4DoF*> vpVertices(nPoses);
                    for(std::size_t i = 0; i < nPoses; i++)
                    {
                        VertexPose4DoF* V4DoF = new VertexPose4DoF(problem.poses[i]);
                        if(problem.fixed[i])
                            V4DoF->setFixed(true);
                        V4DoF->setId(static_cast<int>(i));
                        V4DoF->setMarginalized(false);
                        optimizer.addVertex(V4DoF);
                        vpVertices[i] = V4DoF;
                    }

                    for(std::size_t k = 0; k < problem.constraints(); k++)
                    {
                        Eigen::Matrix4d Tij = Eigen::Matrix4d::Identity();
                        Tij.block<3, 3>(0, 0) = problem.Rij[k];
                        Tij.block<3, 1>(0, 3) = problem.tij[k];

                        Edge4DoF* e = new Edge4DoF(Tij);
                        e->setVertex(1, vpVertices[problem.to[k]]);
                        e->setVertex(0, vpVertices[problem.from[k]]);
                        e->information() = problem.information;
                        optimizer.addEdge(e);
                    }

                    optimizer.initializeOptimization();
                    optimizer.optimize(options.nIterations);

                    for(std::size_t i = 0; i < nPoses; i++)
                    {
                        const ImuCamPose &found = vpVertices[i]->estimate();
                        BodyPose &pose = problem.poses[i];
                        pose.Rwb = found.Rwb;
                        pose.twb = found.twb;
                        for(int c = 0; c < pose.nCameras; c++)
                        {
                            pose.Rcw[c] = found.Rcw[c];
                            pose.tcw[c] = found.tcw[c];
                        }
                    }
                }
            };

        } // namespace

        std::unique_ptr<Pose4DofGraphSolver> MakePose4DofGraphSolver()
        {
            return std::make_unique<G2oPose4DofGraphSolver>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
