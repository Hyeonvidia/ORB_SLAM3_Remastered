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
#include <cstring>
#include <memory>
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
    namespace
    {
        // Member by member: the constructor from the three would normalise a
        // rotation that already is.
        g2o::Sim3 MakeSim3(const Eigen::Quaterniond &R, const Eigen::Vector3d &t, double s)
        {
            g2o::Sim3 S;
            S.rotation() = R;
            S.translation() = t;
            S.scale() = s;
            return S;
        }

        // The poses of an essential graph as they are gathered, and then laid
        // out in the order of the keyframes' ids -- the order v1.0's graph had
        // them in, the id of a vertex being the keyframe's. A keyframe given
        // twice is the one given first, as it was there.
        struct GraphPoses
        {
            struct Pose
            {
                KeyFrame* pKF;
                g2o::Sim3 Siw;
                bool bFixed;
                bool bFixScale;
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            };
            std::vector<Pose, Eigen::aligned_allocator<Pose>> vPoses;

            void Add(KeyFrame* pKF, const g2o::Sim3 &Siw, bool bFixed, bool bFixScale)
            {
                vPoses.push_back({pKF, Siw, bFixed, bFixScale});
            }

            void LayOut(optim::Sim3GraphProblem &problem, std::vector<KeyFrame*> &vpPoseKF,
                        std::vector<int> &vnPoseOfId)
            {
                std::stable_sort(vPoses.begin(), vPoses.end(),
                                 [](const Pose &a, const Pose &b) { return a.pKF->mnId < b.pKF->mnId; });
                for(const Pose &pose : vPoses)
                {
                    if(vnPoseOfId[pose.pKF->mnId] != -1)
                        continue;
                    vnPoseOfId[pose.pKF->mnId] = problem.addPose(pose.Siw.rotation(), pose.Siw.translation(),
                                                                 pose.Siw.scale(), pose.bFixed, pose.bFixScale);
                    vpPoseKF.push_back(pose.pKF);
                }
            }
        };

        // A constraint between two keyframes by id; none if either has no
        // pose in the problem, or they are the same keyframe -- v1.0's graph
        // refused those edges.
        void Constrain(optim::Sim3GraphProblem &problem, const std::vector<int> &vnPoseOfId, long unsigned int nIDi,
                       long unsigned int nIDj, const g2o::Sim3 &Sji)
        {
            if(nIDi >= vnPoseOfId.size() || nIDj >= vnPoseOfId.size())
                return;
            const int i = vnPoseOfId[nIDi];
            const int j = vnPoseOfId[nIDj];
            if(i < 0 || j < 0 || i == j)
                return;
            problem.addConstraint(i, j, Sji.rotation(), Sji.translation(), Sji.scale());
        }

        Digest GraphDigest(const optim::Sim3GraphProblem &problem, const std::vector<KeyFrame*> &vpPoseKF)
        {
            Digest digest;
            for(std::size_t i = 0; i < problem.poses(); i++)
                digest.Add('V', vpPoseKF[i]->mnId, problem.R[i].x(), problem.R[i].y(), problem.R[i].z(),
                           problem.R[i].w(), problem.t[i].x(), problem.t[i].y(), problem.t[i].z(), problem.s[i],
                           problem.fixed[i] != 0, problem.fixScale[i] != 0);
            for(std::size_t k = 0; k < problem.constraints(); k++)
                digest.Add('E', k, vpPoseKF[problem.from[k]]->mnId, vpPoseKF[problem.to[k]]->mnId, problem.Rji[k].x(),
                           problem.Rji[k].y(), problem.Rji[k].z(), problem.Rji[k].w(), problem.tji[k].x(),
                           problem.tji[k].y(), problem.tji[k].z(), problem.sji[k]);
            return digest;
        }

        void SolveGraph(optim::Sim3GraphProblem &problem, optim::Sim3GraphSolver &solver)
        {
            optim::SolveOptions options;
            options.nIterations = 20;
            options.damping = optim::SolveOptions::kValue;
            options.dampingValue = 1e-16;
            solver.Solve(problem, options);
        }
    } // namespace

    void EssentialGraphTask::Build(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                   const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                   const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections,
                                   const bool bFixScale)
    {
        mvpKFs = pMap->GetAllKeyFrames();
        mvpMPs = pMap->GetAllMapPoints();
        const std::vector<KeyFrame*> &vpKFs = mvpKFs;

        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        mvScw.resize(nMaxKFid + 1);
        mvnPoseOfId.assign(nMaxKFid + 1, -1);
        Sim3Vector &vScw = mvScw;

        const int minFeat = 100;

        // Set KeyFrame vertices
        GraphPoses poses;
        for(size_t i = 0, iend = vpKFs.size(); i < iend; i++)
        {
            KeyFrame* pKF = vpKFs[i];
            if(pKF->isBad())
                continue;

            const int nIDi = pKF->mnId;

            KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

            if(it != CorrectedSim3.end())
            {
                vScw[nIDi] = it->second;
            }
            else
            {
                Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
                g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);
                vScw[nIDi] = Siw;
            }

            poses.Add(pKF, vScw[nIDi], pKF->mnId == pMap->GetInitKFid(), bFixScale);
        }
        poses.LayOut(mProblem, mvpPoseKF, mvnPoseOfId);

        std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;

        // Set Loop edges
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

                Constrain(mProblem, mvnPoseOfId, nIDi, nIDj, Sji);
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

                Constrain(mProblem, mvnPoseOfId, nIDi, nIDj, Sji);
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
                    Constrain(mProblem, mvnPoseOfId, nIDi, pLKF->mnId, Sli);
                }
            }

            // Covisibility graph edges
            const std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for(std::vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
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

                        Constrain(mProblem, mvnPoseOfId, nIDi, pKFn->mnId, Sni);
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
                Constrain(mProblem, mvnPoseOfId, nIDi, pKF->mPrevKF->mnId, Spi);
            }
        }
    }

    void EssentialGraphTask::Solve(optim::Sim3GraphSolver &solver)
    {
        SolveGraph(mProblem, solver);
    }

    Digest EssentialGraphTask::Input() const
    {
        return GraphDigest(mProblem, mvpPoseKF);
    }

    void EssentialGraphTask::Write(KeyFrame* pCurKF, Written* pPreview) const
    {
        const std::vector<KeyFrame*> &vpKFs = mvpKFs;
        const std::vector<MapPoint*> &vpMPs = mvpMPs;
        const Sim3Vector &vScw = mvScw;
        Sim3Vector vCorrectedSwc(vScw.size());
        if(pPreview)
        {
            pPreview->vTiw.resize(vpKFs.size());
            pPreview->vXw.resize(vpMPs.size());
            pPreview->vbPoint.assign(vpMPs.size(), 0);
        }

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];

            const int nIDi = pKFi->mnId;
            const int n = mvnPoseOfId[nIDi];
            if(n < 0)
                continue;

            g2o::Sim3 CorrectedSiw = MakeSim3(mProblem.R[n], mProblem.t[n], mProblem.s[n]);
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
            double s = CorrectedSiw.scale();

            Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
            if(pPreview)
                pPreview->vTiw[i] = Tiw;
            else
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
            if(pPreview)
            {
                pPreview->vXw[i] = eigCorrectedP3Dw.cast<float>();
                pPreview->vbPoint[i] = 1;
                continue;
            }
            pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

            pMP->UpdateNormalAndDepth();
        }
    }

    void EssentialGraphTask::Apply(Map* pMap, KeyFrame* pCurKF) const
    {
        std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

        Write(pCurKF, nullptr);

        // TODO Check this changeindex
        pMap->IncreaseChangeIndex();
    }

    EssentialGraphTask::Written EssentialGraphTask::Preview(KeyFrame* pCurKF) const
    {
        Written written;
        Write(pCurKF, &written);
        return written;
    }

    bool EssentialGraphTask::Matches(const Written &written) const
    {
        for(size_t i = 0; i < mvpKFs.size(); i++)
        {
            if(mvnPoseOfId[mvpKFs[i]->mnId] < 0)
                continue;
            const Sophus::SE3f Tmap = mvpKFs[i]->GetPose();
            if(std::memcmp(written.vTiw[i].data(), Tmap.data(), 7 * sizeof(float)) != 0)
                return false;
        }
        for(size_t i = 0; i < mvpMPs.size(); i++)
        {
            if(!written.vbPoint[i])
                continue;
            const Eigen::Vector3f Xmap = mvpMPs[i]->GetWorldPos();
            if(std::memcmp(written.vXw[i].data(), Xmap.data(), 3 * sizeof(float)) != 0)
                return false;
        }
        return true;
    }

    void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                           const KeyFrameAndPose &NonCorrectedSim3,
                                           const KeyFrameAndPose &CorrectedSim3,
                                           const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections,
                                           const bool &bFixScale)
    {
#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::OptimizeEssentialGraph(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections,
                                       bFixScale);
#else
        EssentialGraphTask task;
        task.Build(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale);
        const std::unique_ptr<optim::Sim3GraphSolver> pSolver = optim::MakeSim3GraphSolver();
        task.Solve(*pSolver);
        task.Apply(pMap, pCurKF);
#endif
    }

    void MergeGraphTask::Build(KeyFrame* pCurKF, const std::vector<KeyFrame*> &vpFixedKFs,
                               const std::vector<KeyFrame*> &vpFixedCorrectedKFs,
                               const std::vector<KeyFrame*> &vpNonFixedKFs)
    {
        Map* pMap = pCurKF->GetMap();
        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        Sim3Vector vScw(nMaxKFid + 1);
        Sim3Vector vCorrectedSwc(nMaxKFid + 1);
        mvnPoseOfId.assign(nMaxKFid + 1, -1);

        std::vector<bool> vpGoodPose(nMaxKFid + 1);
        mvbBadPose.assign(nMaxKFid + 1, false);
        std::vector<bool> &vpBadPose = mvbBadPose;

        const int minFeat = 100;

        GraphPoses poses;
        for(KeyFrame* pKFi : vpFixedKFs)
        {
            if(pKFi->isBad())
                continue;

            const int nIDi = pKFi->mnId;

            Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

            vCorrectedSwc[nIDi] = Siw.inverse();
            poses.Add(pKFi, Siw, true, true);

            vpGoodPose[nIDi] = true;
            vpBadPose[nIDi] = false;
        }

        std::set<unsigned long> sIdKF;
        for(KeyFrame* pKFi : vpFixedCorrectedKFs)
        {
            if(pKFi->isBad())
                continue;

            const int nIDi = pKFi->mnId;

            Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

            vCorrectedSwc[nIDi] = Siw.inverse();
            poses.Add(pKFi, Siw, true, false);

            Sophus::SE3d Tcw_bef = pKFi->mTcwBefMerge.cast<double>();
            vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(), Tcw_bef.translation(), 1.0);

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

            Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

            vScw[nIDi] = Siw;
            poses.Add(pKFi, Siw, false, false);

            sIdKF.insert(nIDi);

            vpGoodPose[nIDi] = false;
            vpBadPose[nIDi] = true;
        }
        poses.LayOut(mProblem, mvpPoseKF, mvnPoseOfId);

        std::vector<KeyFrame*> vpKFs;
        vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
        vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
        vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
        vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
        std::set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

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

                    Constrain(mProblem, mvnPoseOfId, nIDi, nIDj, Sji);
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
                        Constrain(mProblem, mvnPoseOfId, nIDi, pLKF->mnId, Sli);
                        num_connections++;
                    }
                }
            }

            // Covisibility graph edges
            const std::vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
            for(std::vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
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

                            Constrain(mProblem, mvnPoseOfId, nIDi, pKFn->mnId, Sni);
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
    }

    void MergeGraphTask::Solve(optim::Sim3GraphSolver &solver)
    {
        SolveGraph(mProblem, solver);
    }

    Digest MergeGraphTask::Input() const
    {
        return GraphDigest(mProblem, mvpPoseKF);
    }

    std::vector<Sophus::SE3f> MergeGraphTask::Preview(const std::vector<KeyFrame*> &vpNonFixedKFs) const
    {
        std::vector<Sophus::SE3f> vTiw(vpNonFixedKFs.size());
        for(std::size_t i = 0; i < vpNonFixedKFs.size(); i++)
        {
            KeyFrame* pKFi = vpNonFixedKFs[i];
            if(pKFi->mnId >= mvnPoseOfId.size())
                continue;
            const int n = mvnPoseOfId[pKFi->mnId];
            if(n < 0)
                continue;
            g2o::Sim3 CorrectedSiw = MakeSim3(mProblem.R[n], mProblem.t[n], mProblem.s[n]);
            double s = CorrectedSiw.scale();
            Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation() / s);
            vTiw[i] = Tiw.cast<float>();
        }
        return vTiw;
    }

    bool MergeGraphTask::Matches(const std::vector<KeyFrame*> &vpNonFixedKFs,
                                 const std::vector<Sophus::SE3f> &vTiw) const
    {
        for(std::size_t i = 0; i < vpNonFixedKFs.size(); i++)
        {
            KeyFrame* pKFi = vpNonFixedKFs[i];
            if(pKFi->isBad() || pKFi->mnId >= mvnPoseOfId.size() || mvnPoseOfId[pKFi->mnId] < 0)
                continue;
            const Sophus::SE3f Tmap = pKFi->GetPose();
            if(std::memcmp(vTiw[i].data(), Tmap.data(), 7 * sizeof(float)) != 0)
                return false;
        }
        return true;
    }

    void MergeGraphTask::Apply(KeyFrame* pCurKF, const std::vector<KeyFrame*> &vpNonFixedKFs,
                               const std::vector<MapPoint*> &vpNonCorrectedMPs) const
    {
        Map* pMap = pCurKF->GetMap();
        const std::vector<bool> &vpBadPose = mvbBadPose;

        std::lock_guard<std::mutex> lock(pMap->mMutexMapUpdate);

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        const std::vector<Sophus::SE3f> vTiw = Preview(vpNonFixedKFs);
        for(std::size_t i = 0; i < vpNonFixedKFs.size(); i++)
        {
            KeyFrame* pKFi = vpNonFixedKFs[i];
            if(pKFi->isBad())
                continue;
            if(pKFi->mnId >= mvnPoseOfId.size() || mvnPoseOfId[pKFi->mnId] < 0)
                continue;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(vTiw[i]);
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

    void Optimizer::OptimizeEssentialGraph(KeyFrame* pCurKF, std::vector<KeyFrame*> &vpFixedKFs,
                                           std::vector<KeyFrame*> &vpFixedCorrectedKFs,
                                           std::vector<KeyFrame*> &vpNonFixedKFs,
                                           std::vector<MapPoint*> &vpNonCorrectedMPs)
    {
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

#ifdef ORBSLAM3R_OPT_SHADOW
        shadow::OptimizeEssentialGraph(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs);
#else
        MergeGraphTask task;
        task.Build(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs);
        const std::unique_ptr<optim::Sim3GraphSolver> pSolver = optim::MakeSim3GraphSolver();
        task.Solve(*pSolver);
        task.Apply(pCurKF, vpNonFixedKFs, vpNonCorrectedMPs);
#endif
    }

    void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                               const KeyFrameAndPose &NonCorrectedSim3,
                                               const KeyFrameAndPose &CorrectedSim3,
                                               const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections)
    {
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<4, 4>> BlockSolver_4_4;

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);

        auto* solver = orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kEigen>();

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

            optimizer.addVertex(V4DoF);
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
                optimizer.addEdge(e);

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
                optimizer.addEdge(e);
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
                optimizer.addEdge(e);
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
                    optimizer.addEdge(e);
                }
            }

            // 1.3 Covisibility graph edges
            const std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for(std::vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
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
                        optimizer.addEdge(e);
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

} // namespace ORB_SLAM3
