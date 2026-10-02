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

// See optimization/Shadow.hpp. Nothing of this file is in a build without
// ORBSLAM3R_OPT_SHADOW.
#ifdef ORBSLAM3R_OPT_SHADOW

#include "optimization/Optimizer.hpp"
#include "optimization/Digest.hpp"
#include "optimization/EssentialGraphTask.hpp"
#include "optimization/Shadow.hpp"
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

#include <memory>

namespace ORB_SLAM3
{

    namespace shadow
    {

        // v1.0's Optimizer::OptimizeEssentialGraph after a loop as it was, but
        // that it says what it read of the map.
        void OptimizeEssentialGraphV1(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                      const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                      const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections,
                                      const bool &bFixScale, Digest &read)
        {
            std::size_t nRead = 0; // constraints, in the order their edges are added
            auto NotePose = [&read](bool bAdded, g2o::VertexSim3Expmap* pVertex)
            {
                if(!bAdded)
                    return;
                const g2o::Sim3 &S = pVertex->estimate();
                read.Add('V', static_cast<long unsigned int>(pVertex->id()), S.rotation().x(), S.rotation().y(),
                         S.rotation().z(), S.rotation().w(), S.translation().x(), S.translation().y(),
                         S.translation().z(), S.scale(), pVertex->fixed(), pVertex->_fix_scale);
            };
            auto NoteEdge =
                [&read, &nRead](bool bAdded, long unsigned int from, long unsigned int to, const g2o::Sim3 &S)
            {
                if(!bAdded)
                    return;
                read.Add('E', nRead++, from, to, S.rotation().x(), S.rotation().y(), S.rotation().z(), S.rotation().w(),
                         S.translation().x(), S.translation().y(), S.translation().z(), S.scale());
            };

            // Setup optimizer
            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);
            auto* solver =
                orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_7_3, orbslam3r::g2o_ext::LinearSolver::kEigen>();

            solver->setUserLambdaInit(1e-16);
            optimizer.setAlgorithm(solver);

            const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
            const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

            const unsigned int nMaxKFid = pMap->GetMaxKFid();

            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
            std::vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid + 1);

            std::vector<Eigen::Vector3d> vZvectors(nMaxKFid + 1); // For debugging
            Eigen::Vector3d z_vec;
            z_vec << 0.0, 0.0, 1.0;

            const int minFeat = 100;

            // Set KeyFrame vertices
            for(size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                KeyFrame* pKF = vpKFs[i];
                if(pKF->isBad())
                    continue;
                g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

                const int nIDi = pKF->mnId;

                KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

                if(it != CorrectedSim3.end())
                {
                    vScw[nIDi] = it->second;
                    VSim3->setEstimate(it->second);
                }
                else
                {
                    Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
                    g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);
                    vScw[nIDi] = Siw;
                    VSim3->setEstimate(Siw);
                }

                if(pKF->mnId == pMap->GetInitKFid())
                    VSim3->setFixed(true);

                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);
                VSim3->_fix_scale = bFixScale;

                NotePose(optimizer.addVertex(VSim3), VSim3);
                vZvectors[nIDi] = vScw[nIDi].rotation() * z_vec; // For debugging

                vpVertices[nIDi] = VSim3;
            }

            std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;

            const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

            // Set Loop edges
            int count_loop = 0;
            for(std::map<KeyFrame*, std::set<KeyFrame*>>::const_iterator mit = LoopConnections.begin(),
                                                                         mend = LoopConnections.end();
                mit != mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                const long unsigned int nIDi = pKF->mnId;
                const std::set<KeyFrame*> &spConnections = mit->second;
                const g2o::Sim3 Siw = vScw[nIDi];
                const g2o::Sim3 Swi = Siw.inverse();

                for(std::set<KeyFrame*>::const_iterator sit = spConnections.begin(), send = spConnections.end();
                    sit != send; sit++)
                {
                    const long unsigned int nIDj = (*sit)->mnId;
                    if((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                        continue;

                    const g2o::Sim3 Sjw = vScw[nIDj];
                    const g2o::Sim3 Sji = Sjw * Swi;

                    g2o::EdgeSim3* e = new g2o::EdgeSim3();
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setMeasurement(Sji);

                    e->information() = matLambda;

                    NoteEdge(optimizer.addEdge(e), nIDi, nIDj, Sji);
                    count_loop++;
                    sInsertedEdges.insert(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj)));
                }
            }

            // Set normal edges
            for(size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                KeyFrame* pKF = vpKFs[i];

                const int nIDi = pKF->mnId;

                g2o::Sim3 Swi;

                KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

                if(iti != NonCorrectedSim3.end())
                    Swi = (iti->second).inverse();
                else
                    Swi = vScw[nIDi].inverse();

                KeyFrame* pParentKF = pKF->GetParent();

                // Spanning tree edge
                if(pParentKF)
                {
                    int nIDj = pParentKF->mnId;

                    g2o::Sim3 Sjw;

                    KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                    if(itj != NonCorrectedSim3.end())
                        Sjw = itj->second;
                    else
                        Sjw = vScw[nIDj];

                    g2o::Sim3 Sji = Sjw * Swi;

                    g2o::EdgeSim3* e = new g2o::EdgeSim3();
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setMeasurement(Sji);
                    e->information() = matLambda;
                    NoteEdge(optimizer.addEdge(e), nIDi, nIDj, Sji);
                }

                // Loop edges
                const std::set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
                for(std::set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send;
                    sit++)
                {
                    KeyFrame* pLKF = *sit;
                    if(pLKF->mnId < pKF->mnId)
                    {
                        g2o::Sim3 Slw;

                        KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                        if(itl != NonCorrectedSim3.end())
                            Slw = itl->second;
                        else
                            Slw = vScw[pLKF->mnId];

                        g2o::Sim3 Sli = Slw * Swi;
                        g2o::EdgeSim3* el = new g2o::EdgeSim3();
                        el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                        el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        el->setMeasurement(Sli);
                        el->information() = matLambda;
                        NoteEdge(optimizer.addEdge(el), nIDi, pLKF->mnId, Sli);
                    }
                }

                // Covisibility graph edges
                const std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
                for(std::vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end();
                    vit++)
                {
                    KeyFrame* pKFn = *vit;
                    if(pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
                    {
                        if(!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                        {
                            if(sInsertedEdges.count(
                                   std::make_pair(std::min(pKF->mnId, pKFn->mnId), std::max(pKF->mnId, pKFn->mnId))))
                                continue;

                            g2o::Sim3 Snw;

                            KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                            if(itn != NonCorrectedSim3.end())
                                Snw = itn->second;
                            else
                                Snw = vScw[pKFn->mnId];

                            g2o::Sim3 Sni = Snw * Swi;

                            g2o::EdgeSim3* en = new g2o::EdgeSim3();
                            en->setVertex(1,
                                          dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                            en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                            en->setMeasurement(Sni);
                            en->information() = matLambda;
                            NoteEdge(optimizer.addEdge(en), nIDi, pKFn->mnId, Sni);
                        }
                    }
                }

                // Inertial edges if inertial
                if(pKF->bImu && pKF->mPrevKF)
                {
                    g2o::Sim3 Spw;
                    KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
                    if(itp != NonCorrectedSim3.end())
                        Spw = itp->second;
                    else
                        Spw = vScw[pKF->mPrevKF->mnId];

                    g2o::Sim3 Spi = Spw * Swi;
                    g2o::EdgeSim3* ep = new g2o::EdgeSim3();
                    ep->setVertex(1,
                                  dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
                    ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    ep->setMeasurement(Spi);
                    ep->information() = matLambda;
                    NoteEdge(optimizer.addEdge(ep), nIDi, pKF->mPrevKF->mnId, Spi);
                }
            }

            optimizer.initializeOptimization();
            optimizer.computeActiveErrors();
            optimizer.optimize(20);
            optimizer.computeActiveErrors();
            std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

            // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
            for(size_t i = 0; i < vpKFs.size(); i++)
            {
                KeyFrame* pKFi = vpKFs[i];

                const int nIDi = pKFi->mnId;

                g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
                g2o::Sim3 CorrectedSiw = VSim3->estimate();
                vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
                double s = CorrectedSiw.scale();

                Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
                pKFi->SetPose(Tiw);
            }

            // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
            for(size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                int nIDr;
                if(pMP->mnCorrectedByKF == pCurKF->mnId)
                {
                    nIDr = pMP->mnCorrectedReference;
                }
                else
                {
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                    nIDr = pRefKF->mnId;
                }

                g2o::Sim3 Srw = vScw[nIDr];
                g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

                Eigen::Matrix<double, 3, 1> eigP3Dw = pMP->GetWorldPos().cast<double>();
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
                pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMP->UpdateNormalAndDepth();
            }

            // TODO Check this changeindex
            pMap->IncreaseChangeIndex();
        }

        void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                    const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                    const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections,
                                    const bool &bFixScale)
        {
            EssentialGraphTask task;
            task.Build(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale);
            const Digest input = task.Input();
            const std::unique_ptr<optim::Sim3GraphSolver> pSolver = optim::MakeSim3GraphSolver();
            task.Solve(*pSolver);
            const EssentialGraphTask::Written written = task.Preview(pCurKF);

            Digest read;
            OptimizeEssentialGraphV1(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale,
                                     read);

            if(read != input)
                Moved("OptimizeEssentialGraph");
            else
                Count("OptimizeEssentialGraph", task.Matches(written));
        }

        // v1.0's Optimizer::OptimizeEssentialGraph after a merge, likewise.
        void OptimizeMergeGraphV1(KeyFrame* pCurKF, std::vector<KeyFrame*> &vpFixedKFs,
                                  std::vector<KeyFrame*> &vpFixedCorrectedKFs, std::vector<KeyFrame*> &vpNonFixedKFs,
                                  std::vector<MapPoint*> &vpNonCorrectedMPs, Digest &read)
        {
            std::size_t nRead = 0; // constraints, in the order their edges are added
            auto NotePose = [&read](bool bAdded, g2o::VertexSim3Expmap* pVertex)
            {
                if(!bAdded)
                    return;
                const g2o::Sim3 &S = pVertex->estimate();
                read.Add('V', static_cast<long unsigned int>(pVertex->id()), S.rotation().x(), S.rotation().y(),
                         S.rotation().z(), S.rotation().w(), S.translation().x(), S.translation().y(),
                         S.translation().z(), S.scale(), pVertex->fixed(), pVertex->_fix_scale);
            };
            auto NoteEdge =
                [&read, &nRead](bool bAdded, long unsigned int from, long unsigned int to, const g2o::Sim3 &S)
            {
                if(!bAdded)
                    return;
                read.Add('E', nRead++, from, to, S.rotation().x(), S.rotation().y(), S.rotation().z(), S.rotation().w(),
                         S.translation().x(), S.translation().y(), S.translation().z(), S.scale());
            };

            Verbose::PrintMess("Opt_Essential: There are " + std::to_string(vpFixedKFs.size()) +
                                   " KFs fixed in the merged map",
                               Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("Opt_Essential: There are " + std::to_string(vpFixedCorrectedKFs.size()) +
                                   " KFs fixed in the old map",
                               Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("Opt_Essential: There are " + std::to_string(vpNonFixedKFs.size()) +
                                   " KFs non-fixed in the merged map",
                               Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("Opt_Essential: There are " + std::to_string(vpNonCorrectedMPs.size()) +
                                   " MPs non-corrected in the merged map",
                               Verbose::VERBOSITY_DEBUG);

            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);
            auto* solver =
                orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_7_3, orbslam3r::g2o_ext::LinearSolver::kEigen>();

            solver->setUserLambdaInit(1e-16);
            optimizer.setAlgorithm(solver);

            Map* pMap = pCurKF->GetMap();
            const unsigned int nMaxKFid = pMap->GetMaxKFid();

            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
            std::vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid + 1);

            std::vector<bool> vpGoodPose(nMaxKFid + 1);
            std::vector<bool> vpBadPose(nMaxKFid + 1);

            const int minFeat = 100;

            for(KeyFrame* pKFi : vpFixedKFs)
            {
                if(pKFi->isBad())
                    continue;

                g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

                const int nIDi = pKFi->mnId;

                Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
                g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

                vCorrectedSwc[nIDi] = Siw.inverse();
                VSim3->setEstimate(Siw);

                VSim3->setFixed(true);

                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);
                VSim3->_fix_scale = true;

                NotePose(optimizer.addVertex(VSim3), VSim3);

                vpVertices[nIDi] = VSim3;

                vpGoodPose[nIDi] = true;
                vpBadPose[nIDi] = false;
            }
            Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

            std::set<unsigned long> sIdKF;
            for(KeyFrame* pKFi : vpFixedCorrectedKFs)
            {
                if(pKFi->isBad())
                    continue;

                g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

                const int nIDi = pKFi->mnId;

                Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
                g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

                vCorrectedSwc[nIDi] = Siw.inverse();
                VSim3->setEstimate(Siw);

                Sophus::SE3d Tcw_bef = pKFi->mTcwBefMerge.cast<double>();
                vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(), Tcw_bef.translation(), 1.0);

                VSim3->setFixed(true);

                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);

                NotePose(optimizer.addVertex(VSim3), VSim3);

                vpVertices[nIDi] = VSim3;

                sIdKF.insert(nIDi);

                vpGoodPose[nIDi] = true;
                vpBadPose[nIDi] = true;
            }

            for(KeyFrame* pKFi : vpNonFixedKFs)
            {
                if(pKFi->isBad())
                    continue;

                const int nIDi = pKFi->mnId;

                if(sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
                    continue;

                g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

                Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
                g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);

                VSim3->setFixed(false);

                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);

                NotePose(optimizer.addVertex(VSim3), VSim3);

                vpVertices[nIDi] = VSim3;

                sIdKF.insert(nIDi);

                vpGoodPose[nIDi] = false;
                vpBadPose[nIDi] = true;
            }

            std::vector<KeyFrame*> vpKFs;
            vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
            vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
            vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
            vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
            std::set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

            const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

            for(KeyFrame* pKFi : vpKFs)
            {
                int num_connections = 0;
                const int nIDi = pKFi->mnId;

                g2o::Sim3 correctedSwi;
                g2o::Sim3 Swi;

                if(vpGoodPose[nIDi])
                    correctedSwi = vCorrectedSwc[nIDi];
                if(vpBadPose[nIDi])
                    Swi = vScw[nIDi].inverse();

                KeyFrame* pParentKFi = pKFi->GetParent();

                // Spanning tree edge
                if(pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
                {
                    int nIDj = pParentKFi->mnId;

                    g2o::Sim3 Sjw;
                    bool bHasRelation = false;

                    if(vpGoodPose[nIDi] && vpGoodPose[nIDj])
                    {
                        Sjw = vCorrectedSwc[nIDj].inverse();
                        bHasRelation = true;
                    }
                    else if(vpBadPose[nIDi] && vpBadPose[nIDj])
                    {
                        Sjw = vScw[nIDj];
                        bHasRelation = true;
                    }

                    if(bHasRelation)
                    {
                        g2o::Sim3 Sji = Sjw * Swi;

                        g2o::EdgeSim3* e = new g2o::EdgeSim3();
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        e->setMeasurement(Sji);

                        e->information() = matLambda;
                        NoteEdge(optimizer.addEdge(e), nIDi, nIDj, Sji);
                        num_connections++;
                    }
                }

                // Loop edges
                const std::set<KeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
                for(std::set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send;
                    sit++)
                {
                    KeyFrame* pLKF = *sit;
                    if(spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId)
                    {
                        g2o::Sim3 Slw;
                        bool bHasRelation = false;

                        if(vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId])
                        {
                            Slw = vCorrectedSwc[pLKF->mnId].inverse();
                            bHasRelation = true;
                        }
                        else if(vpBadPose[nIDi] && vpBadPose[pLKF->mnId])
                        {
                            Slw = vScw[pLKF->mnId];
                            bHasRelation = true;
                        }

                        if(bHasRelation)
                        {
                            g2o::Sim3 Sli = Slw * Swi;
                            g2o::EdgeSim3* el = new g2o::EdgeSim3();
                            el->setVertex(1,
                                          dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                            el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                            el->setMeasurement(Sli);
                            el->information() = matLambda;
                            NoteEdge(optimizer.addEdge(el), nIDi, pLKF->mnId, Sli);
                            num_connections++;
                        }
                    }
                }

                // Covisibility graph edges
                const std::vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
                for(std::vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end();
                    vit++)
                {
                    KeyFrame* pKFn = *vit;
                    if(pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) &&
                       spKFs.find(pKFn) != spKFs.end())
                    {
                        if(!pKFn->isBad() && pKFn->mnId < pKFi->mnId)
                        {
                            g2o::Sim3 Snw = vScw[pKFn->mnId];
                            bool bHasRelation = false;

                            if(vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId])
                            {
                                Snw = vCorrectedSwc[pKFn->mnId].inverse();
                                bHasRelation = true;
                            }
                            else if(vpBadPose[nIDi] && vpBadPose[pKFn->mnId])
                            {
                                Snw = vScw[pKFn->mnId];
                                bHasRelation = true;
                            }

                            if(bHasRelation)
                            {
                                g2o::Sim3 Sni = Snw * Swi;

                                g2o::EdgeSim3* en = new g2o::EdgeSim3();
                                en->setVertex(
                                    1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                                en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                                en->setMeasurement(Sni);
                                en->information() = matLambda;
                                NoteEdge(optimizer.addEdge(en), nIDi, pKFn->mnId, Sni);
                                num_connections++;
                            }
                        }
                    }
                }

                if(num_connections == 0)
                {
                    Verbose::PrintMess("Opt_Essential: KF " + std::to_string(pKFi->mnId) + " has 0 connections",
                                       Verbose::VERBOSITY_DEBUG);
                }
            }

            // Optimize!
            optimizer.initializeOptimization();
            optimizer.optimize(20);

            std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

            // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
            for(KeyFrame* pKFi : vpNonFixedKFs)
            {
                if(pKFi->isBad())
                    continue;

                const int nIDi = pKFi->mnId;

                g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
                g2o::Sim3 CorrectedSiw = VSim3->estimate();
                vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
                double s = CorrectedSiw.scale();
                Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation() / s);

                pKFi->mTcwBefMerge = pKFi->GetPose();
                pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
                pKFi->SetPose(Tiw.cast<float>());
            }

            // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
            for(MapPoint* pMPi : vpNonCorrectedMPs)
            {
                if(pMPi->isBad())
                    continue;

                KeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
                while(pRefKF->isBad())
                {
                    if(!pRefKF)
                    {
                        Verbose::PrintMess("MP " + std::to_string(pMPi->mnId) + " without a valid reference KF",
                                           Verbose::VERBOSITY_DEBUG);
                        break;
                    }

                    pMPi->EraseObservation(pRefKF);
                    pRefKF = pMPi->GetReferenceKeyFrame();
                }

                if(vpBadPose[pRefKF->mnId])
                {
                    Sophus::SE3f TNonCorrectedwr = pRefKF->mTwcBefMerge;
                    Sophus::SE3f Twr = pRefKF->GetPoseInverse();

                    Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
                    pMPi->SetWorldPos(eigCorrectedP3Dw);

                    pMPi->UpdateNormalAndDepth();
                }
                else
                {
                    std::cout << "ERROR: MapPoint has a reference KF from another map" << std::endl;
                }
            }
        }

        void OptimizeEssentialGraph(KeyFrame* pCurKF, std::vector<KeyFrame*> &vpFixedKFs,
                                    std::vector<KeyFrame*> &vpFixedCorrectedKFs, std::vector<KeyFrame*> &vpNonFixedKFs,
                                    std::vector<MapPoint*> &vpNonCorrectedMPs)
        {
            MergeGraphTask task;
            task.Build(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs);
            const Digest input = task.Input();
            const std::unique_ptr<optim::Sim3GraphSolver> pSolver = optim::MakeSim3GraphSolver();
            task.Solve(*pSolver);
            const std::vector<Sophus::SE3f> vTiw = task.Preview(vpNonFixedKFs);

            Digest read;
            OptimizeMergeGraphV1(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs, read);

            if(read != input)
                Moved("OptimizeEssentialGraph (merge)");
            else
                Count("OptimizeEssentialGraph (merge)", task.Matches(vpNonFixedKFs, vTiw));
        }

        // v1.0's Optimizer::OptimizeEssentialGraph4DoF, likewise.
        void OptimizeEssentialGraph4DoFV1(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                          const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                          const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections, Digest &read)
        {
            std::size_t nRead = 0; // constraints, in the order their edges are added
            auto NotePose = [&read](bool bAdded, VertexPose4DoF* pVertex)
            {
                if(!bAdded)
                    return;
                const ImuCamPose &P = pVertex->estimate();
                read.Add('V', static_cast<long unsigned int>(pVertex->id()), P.Rwb, P.twb, P.Rcw[0], P.tcw[0], P.Rcb[0],
                         P.tcb[0], pVertex->fixed());
            };
            auto NoteEdge =
                [&read, &nRead](bool bAdded, long unsigned int from, long unsigned int to, const Eigen::Matrix4d &T)
            {
                if(!bAdded)
                    return;
                const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
                const Eigen::Vector3d t = T.block<3, 1>(0, 3);
                read.Add('E', nRead++, from, to, R, t);
            };
            typedef g2o::BlockSolver<g2o::BlockSolverTraits<4, 4>> BlockSolver_4_4;

            // Setup optimizer
            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);

            auto* solver =
                orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kEigen>();

            optimizer.setAlgorithm(solver);

            const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
            const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

            const unsigned int nMaxKFid = pMap->GetMaxKFid();

            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);

            std::vector<VertexPose4DoF*> vpVertices(nMaxKFid + 1);

            const int minFeat = 100;
            // Set KeyFrame vertices
            for(size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                KeyFrame* pKF = vpKFs[i];
                if(pKF->isBad())
                    continue;

                VertexPose4DoF* V4DoF;

                const int nIDi = pKF->mnId;

                KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

                if(it != CorrectedSim3.end())
                {
                    vScw[nIDi] = it->second;
                    const g2o::Sim3 Swc = it->second.inverse();
                    Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
                    Eigen::Vector3d twc = Swc.translation();
                    V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
                }
                else
                {
                    Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
                    g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

                    vScw[nIDi] = Siw;
                    V4DoF = new VertexPose4DoF(pKF);
                }

                if(pKF == pLoopKF)
                    V4DoF->setFixed(true);

                V4DoF->setId(nIDi);
                V4DoF->setMarginalized(false);

                NotePose(optimizer.addVertex(V4DoF), V4DoF);
                vpVertices[nIDi] = V4DoF;
            }
            std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;

            // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
            Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
            matLambda(0, 0) = 1e3;
            matLambda(1, 1) = 1e3;
            matLambda(0, 0) = 1e3;

            // Set Loop edges
            Edge4DoF* e_loop;
            for(std::map<KeyFrame*, std::set<KeyFrame*>>::const_iterator mit = LoopConnections.begin(),
                                                                         mend = LoopConnections.end();
                mit != mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                const long unsigned int nIDi = pKF->mnId;
                const std::set<KeyFrame*> &spConnections = mit->second;
                const g2o::Sim3 Siw = vScw[nIDi];

                for(std::set<KeyFrame*>::const_iterator sit = spConnections.begin(), send = spConnections.end();
                    sit != send; sit++)
                {
                    const long unsigned int nIDj = (*sit)->mnId;
                    if((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                        continue;

                    const g2o::Sim3 Sjw = vScw[nIDj];
                    const g2o::Sim3 Sij = Siw * Sjw.inverse();
                    Eigen::Matrix4d Tij;
                    Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
                    Tij.block<3, 1>(0, 3) = Sij.translation();
                    Tij(3, 3) = 1.;

                    Edge4DoF* e = new Edge4DoF(Tij);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));

                    e->information() = matLambda;
                    e_loop = e;
                    NoteEdge(optimizer.addEdge(e), nIDi, nIDj, Tij);

                    sInsertedEdges.insert(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj)));
                }
            }

            // 1. Set normal edges
            for(size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                KeyFrame* pKF = vpKFs[i];

                const int nIDi = pKF->mnId;

                g2o::Sim3 Siw;

                // Use noncorrected poses for posegraph edges
                KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

                if(iti != NonCorrectedSim3.end())
                    Siw = iti->second;
                else
                    Siw = vScw[nIDi];

                // 1.1.0 Spanning tree edge
                KeyFrame* pParentKF = static_cast<KeyFrame*>(NULL);
                if(pParentKF)
                {
                    int nIDj = pParentKF->mnId;

                    g2o::Sim3 Swj;

                    KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                    if(itj != NonCorrectedSim3.end())
                        Swj = (itj->second).inverse();
                    else
                        Swj = vScw[nIDj].inverse();

                    g2o::Sim3 Sij = Siw * Swj;
                    Eigen::Matrix4d Tij;
                    Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
                    Tij.block<3, 1>(0, 3) = Sij.translation();
                    Tij(3, 3) = 1.;

                    Edge4DoF* e = new Edge4DoF(Tij);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                    e->information() = matLambda;
                    NoteEdge(optimizer.addEdge(e), nIDi, nIDj, Tij);
                }

                // 1.1.1 Inertial edges
                KeyFrame* prevKF = pKF->mPrevKF;
                if(prevKF)
                {
                    int nIDj = prevKF->mnId;

                    g2o::Sim3 Swj;

                    KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

                    if(itj != NonCorrectedSim3.end())
                        Swj = (itj->second).inverse();
                    else
                        Swj = vScw[nIDj].inverse();

                    g2o::Sim3 Sij = Siw * Swj;
                    Eigen::Matrix4d Tij;
                    Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
                    Tij.block<3, 1>(0, 3) = Sij.translation();
                    Tij(3, 3) = 1.;

                    Edge4DoF* e = new Edge4DoF(Tij);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                    e->information() = matLambda;
                    NoteEdge(optimizer.addEdge(e), nIDi, nIDj, Tij);
                }

                // 1.2 Loop edges
                const std::set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
                for(std::set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send;
                    sit++)
                {
                    KeyFrame* pLKF = *sit;
                    if(pLKF->mnId < pKF->mnId)
                    {
                        g2o::Sim3 Swl;

                        KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                        if(itl != NonCorrectedSim3.end())
                            Swl = itl->second.inverse();
                        else
                            Swl = vScw[pLKF->mnId].inverse();

                        g2o::Sim3 Sil = Siw * Swl;
                        Eigen::Matrix4d Til;
                        Til.block<3, 3>(0, 0) = Sil.rotation().toRotationMatrix();
                        Til.block<3, 1>(0, 3) = Sil.translation();
                        Til(3, 3) = 1.;

                        Edge4DoF* e = new Edge4DoF(Til);
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                        e->information() = matLambda;
                        NoteEdge(optimizer.addEdge(e), nIDi, pLKF->mnId, Til);
                    }
                }

                // 1.3 Covisibility graph edges
                const std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
                for(std::vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end();
                    vit++)
                {
                    KeyFrame* pKFn = *vit;
                    if(pKFn && pKFn != pParentKF && pKFn != prevKF && pKFn != pKF->mNextKF && !pKF->hasChild(pKFn) &&
                       !sLoopEdges.count(pKFn))
                    {
                        if(!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                        {
                            if(sInsertedEdges.count(
                                   std::make_pair(std::min(pKF->mnId, pKFn->mnId), std::max(pKF->mnId, pKFn->mnId))))
                                continue;

                            g2o::Sim3 Swn;

                            KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                            if(itn != NonCorrectedSim3.end())
                                Swn = itn->second.inverse();
                            else
                                Swn = vScw[pKFn->mnId].inverse();

                            g2o::Sim3 Sin = Siw * Swn;
                            Eigen::Matrix4d Tin;
                            Tin.block<3, 3>(0, 0) = Sin.rotation().toRotationMatrix();
                            Tin.block<3, 1>(0, 3) = Sin.translation();
                            Tin(3, 3) = 1.;
                            Edge4DoF* e = new Edge4DoF(Tin);
                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                            e->information() = matLambda;
                            NoteEdge(optimizer.addEdge(e), nIDi, pKFn->mnId, Tin);
                        }
                    }
                }
            }

            optimizer.initializeOptimization();
            optimizer.computeActiveErrors();
            optimizer.optimize(20);

            std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

            // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
            for(size_t i = 0; i < vpKFs.size(); i++)
            {
                KeyFrame* pKFi = vpKFs[i];

                const int nIDi = pKFi->mnId;

                VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
                Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
                Eigen::Vector3d ti = Vi->estimate().tcw[0];

                g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri, ti, 1.);
                vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

                Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation());
                pKFi->SetPose(Tiw.cast<float>());
            }

            // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
            for(size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                int nIDr;

                KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;

                g2o::Sim3 Srw = vScw[nIDr];
                g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

                Eigen::Matrix<double, 3, 1> eigP3Dw = pMP->GetWorldPos().cast<double>();
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
                pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMP->UpdateNormalAndDepth();
            }
            pMap->IncreaseChangeIndex();
        }

        void OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                        const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                        const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections)
        {
            EssentialGraph4DofTask task;
            task.Build(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
            const Digest input = task.Input();
            const std::unique_ptr<optim::Pose4DofGraphSolver> pSolver = optim::MakePose4DofGraphSolver();
            task.Solve(*pSolver);
            const EssentialGraphTask::Written written = task.Preview();

            Digest read;
            OptimizeEssentialGraph4DoFV1(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, read);

            if(read != input)
                Moved("OptimizeEssentialGraph4DoF");
            else
                Count("OptimizeEssentialGraph4DoF", task.Matches(written));
        }

    } // namespace shadow

} // namespace ORB_SLAM3

#endif // ORBSLAM3R_OPT_SHADOW
