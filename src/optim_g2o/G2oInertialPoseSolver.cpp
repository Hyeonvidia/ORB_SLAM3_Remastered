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

#include "optim/InertialPoseSolver.hpp"

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

            // The graph v1.0 built in Optimizer::PoseInertialOptimizationLast-
            // KeyFrame and ...LastFrame, edge for edge and in the same order:
            // pose, velocity and the two biases of the frame, the same four of
            // the earlier state, a unary edge per observation, then the
            // inertial edge, the two bias walks and, when the earlier state is
            // free, its prior. Dense Gauss-Newton.
            class G2oInertialPoseSolver : public InertialPoseSolver
            {
            public:
                void Prepare(const InertialPoseProblem &problem) override
                {
                    mpOptimizer = std::make_unique<g2o::SparseOptimizer>();
                    auto* solver = orbslam3r::g2o_ext::MakeGaussNewton<g2o::BlockSolverX,
                                                                       orbslam3r::g2o_ext::LinearSolver::kDense>();
                    mpOptimizer->setVerbose(false);
                    mpOptimizer->setAlgorithm(solver);

                    // Set Frame vertex
                    mVP = new VertexPose(problem.state.pose);
                    mVP->setId(0);
                    mVP->setFixed(false);
                    mpOptimizer->addVertex(mVP);
                    mVV = new VertexVelocity(problem.state.velocity);
                    mVV->setId(1);
                    mVV->setFixed(false);
                    mpOptimizer->addVertex(mVV);
                    mVG = new VertexGyroBias(problem.state.gyroBias);
                    mVG->setId(2);
                    mVG->setFixed(false);
                    mpOptimizer->addVertex(mVG);
                    mVA = new VertexAccBias(problem.state.accBias);
                    mVA->setId(3);
                    mVA->setFixed(false);
                    mpOptimizer->addVertex(mVA);

                    const std::size_t n = problem.size();
                    mvpMono.assign(n, nullptr);
                    mvpStereo.assign(n, nullptr);
                    for(std::size_t i = 0; i < n; i++)
                    {
                        g2o::OptimizableGraph::Edge* pEdge = nullptr;
                        if(problem.kind[i] == kStereo)
                        {
                            EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(problem.Xw[i].cast<float>());

                            e->setVertex(0, mVP);
                            e->setMeasurement(problem.uv[i]);
                            e->setInformation(Eigen::Matrix3d::Identity() * problem.invSigma2[i]);
                            mvpStereo[i] = e;
                            pEdge = e;
                        }
                        else
                        {
                            EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(problem.Xw[i].cast<float>(),
                                                                       problem.kind[i] == kRight ? 1 : 0);

                            e->setVertex(0, mVP);
                            e->setMeasurement(problem.uv[i].head<2>());
                            e->setInformation(Eigen::Matrix2d::Identity() * problem.invSigma2[i]);
                            mvpMono[i] = e;
                            pEdge = e;
                        }

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        pEdge->setRobustKernel(rk);
                        rk->setDelta(problem.huber[i]);

                        mpOptimizer->addEdge(pEdge);
                    }
                    mvbRobust.assign(n, 1);

                    mbPreviousFixed = problem.previousFixed;
                    mVPk = new VertexPose(problem.previous.pose);
                    mVPk->setId(4);
                    mVPk->setFixed(mbPreviousFixed);
                    mpOptimizer->addVertex(mVPk);
                    mVVk = new VertexVelocity(problem.previous.velocity);
                    mVVk->setId(5);
                    mVVk->setFixed(mbPreviousFixed);
                    mpOptimizer->addVertex(mVVk);
                    mVGk = new VertexGyroBias(problem.previous.gyroBias);
                    mVGk->setId(6);
                    mVGk->setFixed(mbPreviousFixed);
                    mpOptimizer->addVertex(mVGk);
                    mVAk = new VertexAccBias(problem.previous.accBias);
                    mVAk->setId(7);
                    mVAk->setFixed(mbPreviousFixed);
                    mpOptimizer->addVertex(mVAk);

                    mpInertial = new EdgeInertial(problem.preintegration);

                    mpInertial->setVertex(0, mVPk);
                    mpInertial->setVertex(1, mVVk);
                    mpInertial->setVertex(2, mVGk);
                    mpInertial->setVertex(3, mVAk);
                    mpInertial->setVertex(4, mVP);
                    mpInertial->setVertex(5, mVV);
                    mpOptimizer->addEdge(mpInertial);

                    mpGyroWalk = new EdgeGyroRW();
                    mpGyroWalk->setVertex(0, mVGk);
                    mpGyroWalk->setVertex(1, mVG);
                    mpGyroWalk->setInformation(problem.gyroWalkInformation);
                    mpOptimizer->addEdge(mpGyroWalk);

                    mpAccWalk = new EdgeAccRW();
                    mpAccWalk->setVertex(0, mVAk);
                    mpAccWalk->setVertex(1, mVA);
                    mpAccWalk->setInformation(problem.accWalkInformation);
                    mpOptimizer->addEdge(mpAccWalk);

                    mpPrior = nullptr;
                    if(!mbPreviousFixed)
                    {
                        mpPrior = new EdgePriorPoseImu(problem.priorRwb, problem.priorTwb, problem.priorVelocity,
                                                       problem.priorGyroBias, problem.priorAccBias,
                                                       problem.priorInformation);

                        mpPrior->setVertex(0, mVPk);
                        mpPrior->setVertex(1, mVVk);
                        mpPrior->setVertex(2, mVGk);
                        mpPrior->setVertex(3, mVAk);
                        g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
                        mpPrior->setRobustKernel(rkp);
                        rkp->setDelta(problem.priorHuber);
                        mpOptimizer->addEdge(mpPrior);
                    }
                }

                void Solve(InertialPoseProblem &problem, int nIterations) override
                {
                    const std::size_t n = problem.size();
                    for(std::size_t i = 0; i < n; i++)
                    {
                        g2o::OptimizableGraph::Edge* pEdge = Edge(i);
                        pEdge->setLevel(problem.active[i] ? 0 : 1);
                        if(mvbRobust[i] && !problem.robust[i])
                        {
                            pEdge->setRobustKernel(0);
                            mvbRobust[i] = 0;
                        }
                    }

                    mpOptimizer->initializeOptimization(0);
                    mpOptimizer->optimize(nIterations);

                    // An edge that took part has the error the optimiser left
                    // in it; one that did not has none until it is asked for.
                    for(std::size_t i = 0; i < n; i++)
                    {
                        g2o::OptimizableGraph::Edge* pEdge = Edge(i);
                        if(!problem.active[i])
                            pEdge->computeError();
                        problem.chi2[i] = pEdge->chi2();
                        problem.depthPositive[i] = mvpMono[i] ? mvpMono[i]->isDepthPositive() : 1;
                    }
                    Found(problem);
                }

                void Evaluate(InertialPoseProblem &problem) override
                {
                    for(std::size_t i = 0; i < problem.size(); i++)
                    {
                        g2o::OptimizableGraph::Edge* pEdge = Edge(i);
                        pEdge->computeError();
                        problem.chi2[i] = pEdge->chi2();
                    }
                }

                void Information(InertialPoseProblem &problem, const std::vector<unsigned char> &vbInlier) override
                {
                    const std::size_t n = problem.size();
                    if(mbPreviousFixed)
                    {
                        Eigen::Matrix<double, 15, 15> H;
                        H.setZero();

                        H.block<9, 9>(0, 0) += mpInertial->GetHessian2();
                        H.block<3, 3>(9, 9) += mpGyroWalk->GetHessian2();
                        H.block<3, 3>(12, 12) += mpAccWalk->GetHessian2();

                        for(std::size_t i = 0; i < n; i++)
                            if(mvpMono[i] && vbInlier[i])
                                H.block<6, 6>(0, 0) += mvpMono[i]->GetHessian();

                        for(std::size_t i = 0; i < n; i++)
                            if(mvpStereo[i] && vbInlier[i])
                                H.block<6, 6>(0, 0) += mvpStereo[i]->GetHessian();

                        problem.information = H;
                    }
                    else
                    {
                        Eigen::Matrix<double, 30, 30> H;
                        H.setZero();

                        H.block<24, 24>(0, 0) += mpInertial->GetHessian();

                        Eigen::Matrix<double, 6, 6> Hgr = mpGyroWalk->GetHessian();
                        H.block<3, 3>(9, 9) += Hgr.block<3, 3>(0, 0);
                        H.block<3, 3>(9, 24) += Hgr.block<3, 3>(0, 3);
                        H.block<3, 3>(24, 9) += Hgr.block<3, 3>(3, 0);
                        H.block<3, 3>(24, 24) += Hgr.block<3, 3>(3, 3);

                        Eigen::Matrix<double, 6, 6> Har = mpAccWalk->GetHessian();
                        H.block<3, 3>(12, 12) += Har.block<3, 3>(0, 0);
                        H.block<3, 3>(12, 27) += Har.block<3, 3>(0, 3);
                        H.block<3, 3>(27, 12) += Har.block<3, 3>(3, 0);
                        H.block<3, 3>(27, 27) += Har.block<3, 3>(3, 3);

                        H.block<15, 15>(0, 0) += mpPrior->GetHessian();

                        for(std::size_t i = 0; i < n; i++)
                            if(mvpMono[i] && vbInlier[i])
                                H.block<6, 6>(15, 15) += mvpMono[i]->GetHessian();

                        for(std::size_t i = 0; i < n; i++)
                            if(mvpStereo[i] && vbInlier[i])
                                H.block<6, 6>(15, 15) += mvpStereo[i]->GetHessian();

                        problem.information = H;
                    }
                }

            private:
                g2o::OptimizableGraph::Edge* Edge(std::size_t i) const
                {
                    if(mvpMono[i])
                        return mvpMono[i];
                    return mvpStereo[i];
                }

                static void Found(InertialState &state, const VertexPose* VP, const VertexVelocity* VV,
                                  const VertexGyroBias* VG, const VertexAccBias* VA)
                {
                    const ImuCamPose &found = VP->estimate();
                    state.pose.Rwb = found.Rwb;
                    state.pose.twb = found.twb;
                    for(int c = 0; c < state.pose.nCameras; c++)
                    {
                        state.pose.Rcw[c] = found.Rcw[c];
                        state.pose.tcw[c] = found.tcw[c];
                    }
                    state.velocity = VV->estimate();
                    state.gyroBias = VG->estimate();
                    state.accBias = VA->estimate();
                }

                void Found(InertialPoseProblem &problem) const
                {
                    Found(problem.state, mVP, mVV, mVG, mVA);
                    if(!mbPreviousFixed)
                        Found(problem.previous, mVPk, mVVk, mVGk, mVAk);
                }

                std::unique_ptr<g2o::SparseOptimizer> mpOptimizer;
                // The optimiser's, all of them.
                VertexPose* mVP = nullptr;
                VertexVelocity* mVV = nullptr;
                VertexGyroBias* mVG = nullptr;
                VertexAccBias* mVA = nullptr;
                VertexPose* mVPk = nullptr;
                VertexVelocity* mVVk = nullptr;
                VertexGyroBias* mVGk = nullptr;
                VertexAccBias* mVAk = nullptr;
                std::vector<EdgeMonoOnlyPose*> mvpMono;     // by observation; null if it is stereo
                std::vector<EdgeStereoOnlyPose*> mvpStereo; // by observation; null if it is not
                EdgeInertial* mpInertial = nullptr;
                EdgeGyroRW* mpGyroWalk = nullptr;
                EdgeAccRW* mpAccWalk = nullptr;
                EdgePriorPoseImu* mpPrior = nullptr;
                std::vector<unsigned char> mvbRobust;
                bool mbPreviousFixed = true;
            };

        } // namespace

        std::unique_ptr<InertialPoseSolver> MakeInertialPoseSolver()
        {
            return std::make_unique<G2oInertialPoseSolver>();
        }

    } // namespace optim

} // namespace ORB_SLAM3
