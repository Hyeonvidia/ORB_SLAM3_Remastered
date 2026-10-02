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

#include "optim/Sim3Solver.hpp"

#include "optim_g2o/OptimizableTypes.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
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

            // The graph v1.0 built in Optimizer::OptimizeSim3: one Sim3 vertex,
            // two fixed point vertices per pair and an edge from each to it,
            // on a dense Levenberg-Marquardt.
            class G2oSim3Solver : public Sim3Solver
            {
            public:
                void Prepare(const Sim3Problem &problem) override
                {
                    mpOptimizer = std::make_unique<g2o::SparseOptimizer>();
                    mpOptimizer->setAlgorithm(
                        orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX,
                                                          orbslam3r::g2o_ext::LinearSolver::kDense>());

                    // Member by member: the constructor from the three would
                    // normalise a rotation that already is.
                    g2o::Sim3 S12;
                    S12.rotation() = problem.R12;
                    S12.translation() = problem.t12;
                    S12.scale() = problem.s12;

                    mpSim3 = new VertexSim3Expmap();
                    mpSim3->_fix_scale = problem.fixScale;
                    mpSim3->setEstimate(S12);
                    mpSim3->setId(0);
                    mpSim3->setFixed(false);
                    mpSim3->pCamera1 = problem.camera1;
                    mpSim3->pCamera2 = problem.camera2;
                    mpOptimizer->addVertex(mpSim3);

                    const std::size_t n = problem.size();
                    mvpEdges12.clear();
                    mvpEdges21.clear();
                    mvpEdges12.reserve(n);
                    mvpEdges21.reserve(n);
                    mvbRobust.assign(n, 1);
                    for(std::size_t i = 0; i < n; i++)
                    {
                        g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                        vPoint1->setEstimate(problem.X1[i]);
                        vPoint1->setId(static_cast<int>(2 * i + 1));
                        vPoint1->setFixed(true);
                        mpOptimizer->addVertex(vPoint1);

                        g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                        vPoint2->setEstimate(problem.X2[i]);
                        vPoint2->setId(static_cast<int>(2 * (i + 1)));
                        vPoint2->setFixed(true);
                        mpOptimizer->addVertex(vPoint2);

                        // Set edge x1 = S12*X2
                        EdgeSim3ProjectXYZ* e12 = new EdgeSim3ProjectXYZ();
                        e12->setVertex(0, vPoint2);
                        e12->setVertex(1, mpSim3);
                        e12->setMeasurement(problem.uv1[i]);
                        e12->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2_1[i]);

                        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
                        e12->setRobustKernel(rk1);
                        rk1->setDelta(problem.huber);
                        mpOptimizer->addEdge(e12);

                        // Set edge x2 = S21*X1
                        EdgeInverseSim3ProjectXYZ* e21 = new EdgeInverseSim3ProjectXYZ();
                        e21->setVertex(0, vPoint1);
                        e21->setVertex(1, mpSim3);
                        e21->setMeasurement(problem.uv2[i]);
                        e21->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2_2[i]);

                        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
                        e21->setRobustKernel(rk2);
                        rk2->setDelta(problem.huber);
                        mpOptimizer->addEdge(e21);

                        mvpEdges12.push_back(e12);
                        mvpEdges21.push_back(e21);
                    }
                }

                void Solve(Sim3Problem &problem, int nIterations) override
                {
                    const std::size_t n = mvpEdges12.size();
                    for(std::size_t i = 0; i < n; i++)
                    {
                        const int level = problem.active[i] ? 0 : 1;
                        mvpEdges12[i]->setLevel(level);
                        mvpEdges21[i]->setLevel(level);
                        if(mvbRobust[i] && !problem.robust[i])
                        {
                            mvpEdges12[i]->setRobustKernel(0);
                            mvpEdges21[i]->setRobustKernel(0);
                            mvbRobust[i] = 0;
                        }
                    }

                    mpOptimizer->initializeOptimization(0);
                    mpOptimizer->optimize(nIterations);

                    Read(problem, false);
                }

                void Evaluate(Sim3Problem &problem) override { Read(problem, true); }

            private:
                void Read(Sim3Problem &problem, bool bCompute)
                {
                    for(std::size_t i = 0; i < mvpEdges12.size(); i++)
                    {
                        if(!problem.active[i])
                            continue;
                        if(bCompute)
                        {
                            mvpEdges12[i]->computeError();
                            mvpEdges21[i]->computeError();
                        }
                        problem.chi2_1[i] = mvpEdges12[i]->chi2();
                        problem.chi2_2[i] = mvpEdges21[i]->chi2();
                    }

                    const g2o::Sim3 &found = mpSim3->estimate();
                    problem.R12 = found.rotation();
                    problem.t12 = found.translation();
                    problem.s12 = found.scale();
                }

                std::unique_ptr<g2o::SparseOptimizer> mpOptimizer;
                VertexSim3Expmap* mpSim3 = nullptr; // the optimiser's
                std::vector<EdgeSim3ProjectXYZ*> mvpEdges12;
                std::vector<EdgeInverseSim3ProjectXYZ*> mvpEdges21;
                std::vector<unsigned char> mvbRobust;
            };

        } // namespace

        std::unique_ptr<Sim3Solver> MakeSim3Solver()
        {
            return std::make_unique<G2oSim3Solver>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
