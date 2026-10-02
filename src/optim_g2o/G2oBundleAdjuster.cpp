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

#include "optim/BundleAdjuster.hpp"

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

            // The graph v1.0 built for a bundle adjustment: an SE3 vertex per
            // pose, a point vertex per point, marginalised, and a binary edge
            // per observation, on sparse Levenberg-Marquardt.
            //
            // g2o lays its unknowns out by vertex id and sums its edges in the
            // order they were added, so the ids here are the problem's indices
            // -- the poses, then the points -- and the edges are added in the
            // order of the observations.
            class G2oBundleAdjuster : public BundleAdjuster
            {
            public:
                void Prepare(const BaProblem &problem) override
                {
                    mpOptimizer = std::make_unique<g2o::SparseOptimizer>();
                    mpAlgorithm = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3,
                                                                    orbslam3r::g2o_ext::LinearSolver::kEigen>();
                    mpOptimizer->setAlgorithm(mpAlgorithm);
                    mpOptimizer->setVerbose(false);

                    const std::size_t nPoses = problem.poses();
                    mvpPoses.clear();
                    mvpPoses.reserve(nPoses);
                    for(std::size_t i = 0; i < nPoses; i++)
                    {
                        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                        vSE3->setEstimate(g2o::SE3Quat(problem.Rcw[i], problem.tcw[i]));
                        vSE3->setId(static_cast<int>(i));
                        vSE3->setFixed(problem.poseFixed[i]);
                        mpOptimizer->addVertex(vSE3);
                        mvpPoses.push_back(vSE3);
                    }

                    const std::size_t nPoints = problem.points();
                    mvpPoints.clear();
                    mvpPoints.reserve(nPoints);
                    for(std::size_t i = 0; i < nPoints; i++)
                    {
                        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
                        vPoint->setEstimate(problem.Xw[i]);
                        vPoint->setId(static_cast<int>(nPoses + i));
                        vPoint->setMarginalized(true);
                        mpOptimizer->addVertex(vPoint);
                        mvpPoints.push_back(vPoint);
                    }

                    const std::size_t n = problem.observations();
                    mvpEdges.clear();
                    mvpEdges.reserve(n);
                    mvbRobust.assign(n, 0);
                    for(std::size_t i = 0; i < n; i++)
                    {
                        g2o::VertexSBAPointXYZ* vPoint = mvpPoints[problem.point[i]];
                        g2o::VertexSE3Expmap* vSE3 = mvpPoses[problem.pose[i]];
                        const Rig &rig = problem.rigs[problem.poseRig[problem.pose[i]]];

                        g2o::OptimizableGraph::Edge* pEdge = nullptr;
                        if(problem.kind[i] == kStereo)
                        {
                            g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                            e->setVertex(0, vPoint);
                            e->setVertex(1, vSE3);
                            e->setMeasurement(problem.uv[i]);
                            Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * problem.invSigma2[i];
                            e->setInformation(Info);
                            e->fx = rig.fx;
                            e->fy = rig.fy;
                            e->cx = rig.cx;
                            e->cy = rig.cy;
                            e->bf = rig.bf;
                            pEdge = e;
                        }
                        else if(problem.kind[i] == kMono)
                        {
                            EdgeSE3ProjectXYZ* e = new EdgeSE3ProjectXYZ();
                            e->setVertex(0, vPoint);
                            e->setVertex(1, vSE3);
                            e->setMeasurement(problem.uv[i].head<2>());
                            e->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2[i]);
                            e->pCamera = rig.camera;
                            pEdge = e;
                        }
                        else
                        {
                            EdgeSE3ProjectXYZToBody* e = new EdgeSE3ProjectXYZToBody();
                            e->setVertex(0, vPoint);
                            e->setVertex(1, vSE3);
                            e->setMeasurement(problem.uv[i].head<2>());
                            e->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2[i]);
                            e->mTrl = g2o::SE3Quat(rig.Rrl, rig.trl);
                            e->pCamera = rig.camera2;
                            pEdge = e;
                        }

                        if(problem.robust[i])
                        {
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            pEdge->setRobustKernel(rk);
                            rk->setDelta(problem.huber[i]);
                            mvbRobust[i] = 1;
                        }

                        mpOptimizer->addEdge(pEdge);
                        mvpEdges.push_back(pEdge);
                    }
                }

                void Solve(BaProblem &problem, const SolveOptions &options) override
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

                    if(options.pbStop)
                        mpOptimizer->setForceStopFlag(options.pbStop);
                    if(options.damping == SolveOptions::kValue)
                        mpAlgorithm->setUserLambdaInit(options.dampingValue);
                    else if(options.damping == SolveOptions::kFractionOfDiagonal)
                        mpAlgorithm->setInitialDamping(options.dampingValue);
                    if(options.nStallIterations > 0)
                        mpAlgorithm->setStallIterations(options.nStallIterations);

                    mpOptimizer->initializeOptimization(0);
                    mpOptimizer->optimize(options.nIterations);

                    for(std::size_t i = 0; i < n; i++)
                    {
                        problem.chi2[i] = mvpEdges[i]->chi2();
                        if(problem.kind[i] == kStereo)
                            problem.depthPositive[i] = static_cast<g2o::EdgeStereoSE3ProjectXYZ*>(mvpEdges[i])
                                                           ->isDepthPositive();
                        else if(problem.kind[i] == kMono)
                            problem.depthPositive[i] = static_cast<EdgeSE3ProjectXYZ*>(mvpEdges[i])->isDepthPositive();
                        else
                            problem.depthPositive[i] = static_cast<EdgeSE3ProjectXYZToBody*>(mvpEdges[i])
                                                           ->isDepthPositive();
                    }

                    for(std::size_t i = 0; i < mvpPoses.size(); i++)
                    {
                        const g2o::SE3Quat &found = mvpPoses[i]->estimate();
                        problem.Rcw[i] = found.rotation();
                        problem.tcw[i] = found.translation();
                    }
                    for(std::size_t i = 0; i < mvpPoints.size(); i++)
                        problem.Xw[i] = mvpPoints[i]->estimate();
                }

            private:
                std::unique_ptr<g2o::SparseOptimizer> mpOptimizer;
                orbslam3r::g2o_ext::LevenbergStopOnStall* mpAlgorithm = nullptr; // the optimiser's
                std::vector<g2o::VertexSE3Expmap*> mvpPoses;                     // likewise
                std::vector<g2o::VertexSBAPointXYZ*> mvpPoints;
                std::vector<g2o::OptimizableGraph::Edge*> mvpEdges;
                std::vector<unsigned char> mvbRobust;
            };

        } // namespace

        std::unique_ptr<BundleAdjuster> MakeBundleAdjuster()
        {
            return std::make_unique<G2oBundleAdjuster>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
