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

#include "optim/PoseSolver.hpp"

#include "optim_g2o/OptimizableTypes.hpp"

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

            // The graph v1.0 built in Optimizer::PoseOptimization, edge for
            // edge and in the same order: one free SE3 vertex and a unary edge
            // per observation, on a dense 6x6 Levenberg-Marquardt.
            class G2oPoseSolver : public PoseSolver
            {
            public:
                void Prepare(const PoseProblem &problem) override
                {
                    mpOptimizer = std::make_unique<g2o::SparseOptimizer>();
                    mpOptimizer->setAlgorithm(
                        orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3,
                                                          orbslam3r::g2o_ext::LinearSolver::kDense>());

                    mpPose = new g2o::VertexSE3Expmap();
                    mpPose->setEstimate(g2o::SE3Quat(problem.Rcw, problem.tcw));
                    mpPose->setId(0);
                    mpPose->setFixed(false);
                    mpOptimizer->addVertex(mpPose);

                    const std::size_t n = problem.size();
                    mvpEdges.clear();
                    mvpEdges.reserve(n);
                    for(std::size_t i = 0; i < n; i++)
                    {
                        g2o::OptimizableGraph::Edge* pEdge = nullptr;
                        if(problem.kind[i] == PoseProblem::kStereo)
                        {
                            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                            e->setVertex(0, mpPose);
                            e->setMeasurement(problem.uv[i]);
                            Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * problem.invSigma2[i];
                            e->setInformation(Info);
                            e->fx = problem.fx;
                            e->fy = problem.fy;
                            e->cx = problem.cx;
                            e->cy = problem.cy;
                            e->bf = problem.bf;
                            e->Xw = problem.Xw[i];
                            pEdge = e;
                        }
                        else if(problem.kind[i] == PoseProblem::kMono)
                        {
                            EdgeSE3ProjectXYZOnlyPose* e = new EdgeSE3ProjectXYZOnlyPose();
                            e->setVertex(0, mpPose);
                            e->setMeasurement(problem.uv[i].head<2>());
                            e->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2[i]);
                            e->pCamera = problem.camera;
                            e->Xw = problem.Xw[i];
                            pEdge = e;
                        }
                        else
                        {
                            EdgeSE3ProjectXYZOnlyPoseToBody* e = new EdgeSE3ProjectXYZOnlyPoseToBody();
                            e->setVertex(0, mpPose);
                            e->setMeasurement(problem.uv[i].head<2>());
                            e->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2[i]);
                            e->pCamera = problem.camera2;
                            e->Xw = problem.Xw[i];
                            e->mTrl = g2o::SE3Quat(problem.Rrl, problem.trl);
                            pEdge = e;
                        }

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        pEdge->setRobustKernel(rk);
                        rk->setDelta(problem.huber[i]);

                        mpOptimizer->addEdge(pEdge);
                        mvpEdges.push_back(pEdge);
                    }
                    mvbRobust.assign(n, 1);
                }

                void Solve(PoseProblem &problem, int nIterations) override
                {
                    const std::size_t n = mvpEdges.size();
                    for(std::size_t i = 0; i < n; i++)
                    {
                        mvpEdges[i]->setLevel(problem.active[i] ? 0 : 1);
                        if(mvbRobust[i] && !problem.robust[i])
                        {
                            mvpEdges[i]->setRobustKernel(0);
                            mvbRobust[i] = 0;
                        }
                    }

                    mpPose->setEstimate(g2o::SE3Quat(problem.Rcw, problem.tcw));
                    mpOptimizer->initializeOptimization(0);
                    mpOptimizer->optimize(nIterations);

                    // An edge that took part has the error the optimiser left
                    // in it; one that did not has none until it is asked for.
                    for(std::size_t i = 0; i < n; i++)
                    {
                        if(!problem.active[i])
                            mvpEdges[i]->computeError();
                        problem.chi2[i] = mvpEdges[i]->chi2();
                    }

                    const g2o::SE3Quat &found = mpPose->estimate();
                    problem.Rcw = found.rotation();
                    problem.tcw = found.translation();
                }

            private:
                std::unique_ptr<g2o::SparseOptimizer> mpOptimizer;
                g2o::VertexSE3Expmap* mpPose = nullptr; // the optimiser's
                std::vector<g2o::OptimizableGraph::Edge*> mvpEdges;
                std::vector<unsigned char> mvbRobust;
            };

        } // namespace

        std::unique_ptr<PoseSolver> MakePoseSolver()
        {
            return std::make_unique<G2oPoseSolver>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
