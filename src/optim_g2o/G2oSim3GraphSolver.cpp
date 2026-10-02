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

#include "optim/Sim3GraphSolver.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
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

            // Member by member: the constructor from the three would normalise
            // a rotation that already is.
            g2o::Sim3 MakeSim3(const Eigen::Quaterniond &R, const Eigen::Vector3d &t, double s)
            {
                g2o::Sim3 S;
                S.rotation() = R;
                S.translation() = t;
                S.scale() = s;
                return S;
            }

            // The graph v1.0 built for an essential graph: a Sim3 vertex per
            // pose and an EdgeSim3 per constraint, identity information, on
            // sparse Levenberg-Marquardt over 7-dimensional blocks. The ids
            // are the problem's indices and the edges are added in the order
            // of the constraints, as in G2oBundleAdjuster.
            class G2oSim3GraphSolver : public Sim3GraphSolver
            {
            public:
                void Solve(Sim3GraphProblem &problem, const SolveOptions &options) override
                {
                    g2o::SparseOptimizer optimizer;
                    optimizer.setVerbose(false);
                    auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_7_3,
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

                    const std::size_t nPoses = problem.poses();
                    std::vector<g2o::VertexSim3Expmap*> vpVertices(nPoses);
                    for(std::size_t i = 0; i < nPoses; i++)
                    {
                        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();
                        VSim3->setEstimate(MakeSim3(problem.R[i], problem.t[i], problem.s[i]));
                        VSim3->setFixed(problem.fixed[i]);
                        VSim3->setId(static_cast<int>(i));
                        VSim3->setMarginalized(false);
                        VSim3->_fix_scale = problem.fixScale[i];
                        optimizer.addVertex(VSim3);
                        vpVertices[i] = VSim3;
                    }

                    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();
                    for(std::size_t k = 0; k < problem.constraints(); k++)
                    {
                        g2o::EdgeSim3* e = new g2o::EdgeSim3();
                        e->setVertex(1, vpVertices[problem.to[k]]);
                        e->setVertex(0, vpVertices[problem.from[k]]);
                        e->setMeasurement(MakeSim3(problem.Rji[k], problem.tji[k], problem.sji[k]));
                        e->information() = matLambda;
                        optimizer.addEdge(e);
                    }

                    optimizer.initializeOptimization();
                    optimizer.optimize(options.nIterations);

                    for(std::size_t i = 0; i < nPoses; i++)
                    {
                        const g2o::Sim3 &found = vpVertices[i]->estimate();
                        problem.R[i] = found.rotation();
                        problem.t[i] = found.translation();
                        problem.s[i] = found.scale();
                    }
                }
            };

        } // namespace

        std::unique_ptr<Sim3GraphSolver> MakeSim3GraphSolver()
        {
            return std::make_unique<G2oSim3GraphSolver>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
