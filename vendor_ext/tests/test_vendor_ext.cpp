// Verifies that the wrappers in vendor_ext reproduce what ORB-SLAM3 got by
// editing its vendored copies -- without those edits existing.
//
//   ./test_vendor_ext                 run the self-contained checks
//   ./test_vendor_ext path/to/ORBvoc.txt
//                                     also load the real 971k-word vocabulary
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <opencv2/core.hpp>

#include "orbslam3r/dbow2_ext/orb_vocabulary.hpp"
#include "orbslam3r/dbow2_ext/serialization.hpp"
#include "orbslam3r/g2o_ext/compat.hpp"
#include "orbslam3r/g2o_ext/solver_factory.hpp"

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace
{

    int failures = 0;

    void check(bool ok, const char* what, const std::string &detail = "")
    {
        std::printf("  [%s] %-38s %s\n", ok ? " ok " : "FAIL", what, detail.c_str());
        if(!ok)
            ++failures;
    }

    std::vector<cv::Mat> RandomDescriptors(int n, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<cv::Mat> out;
        out.reserve(n);
        for(int i = 0; i < n; ++i)
        {
            cv::Mat d(1, 32, CV_8U);
            for(int j = 0; j < 32; ++j)
                d.at<uint8_t>(0, j) = static_cast<uint8_t>(byte(rng));
            out.push_back(d);
        }
        return out;
    }

    // ---------------------------------------------------------------------------
    // dbow2_ext: the text vocabulary format, as a subclass of untouched upstream.
    void TestTextVocabulary()
    {
        std::puts("dbow2_ext :: TextFileVocabulary");

        orbslam3r::ORBVocabulary voc(3, 3, DBoW2::TF_IDF, DBoW2::L1_NORM);
        std::vector<std::vector<cv::Mat>> training;
        for(int i = 0; i < 12; ++i)
            training.push_back(RandomDescriptors(20, 1000 + i));
        voc.create(training);
        check(voc.size() > 0, "upstream create() still works", std::to_string(voc.size()) + " words");

        const std::string path = "/tmp/orbslam3r_voc_roundtrip.txt";
        check(voc.saveToTextFile(path), "saveToTextFile");

        orbslam3r::ORBVocabulary reloaded;
        check(reloaded.loadFromTextFile(path), "loadFromTextFile");
        check(reloaded.size() == voc.size(), "word count survives the round trip",
              std::to_string(reloaded.size()) + " vs " + std::to_string(voc.size()));
        check(reloaded.getBranchingFactor() == voc.getBranchingFactor() &&
                  reloaded.getDepthLevels() == voc.getDepthLevels(),
              "tree shape survives the round trip");

        // Scoring must be identical, not merely structurally similar: transform the
        // same image twice and compare the BoW vectors.
        const std::vector<cv::Mat> query = RandomDescriptors(30, 77);
        DBoW2::BowVector a, b;
        DBoW2::FeatureVector fa, fb;
        voc.transform(query, a, fa, 4);
        reloaded.transform(query, b, fb, 4);
        check(a.size() == b.size() && voc.score(a, b) > 0.999, "reloaded vocabulary scores identically",
              "score=" + std::to_string(voc.score(a, b)));

        // Defect 1: upstream's guard was `if (f.eof()) return false`, which is false
        // for a stream that never opened, so a bad path reported success.
        orbslam3r::ORBVocabulary missing;
        check(!missing.loadFromTextFile("/tmp/definitely_not_a_vocabulary_9f3a.txt"), "missing file reports failure");

        // Defect 2: `while (!f.eof())` ran one extra iteration on a trailing
        // newline and appended a node with an uninitialised parent id.
        {
            std::ifstream in(path);
            std::stringstream body;
            body << in.rdbuf();
            std::ofstream out("/tmp/orbslam3r_voc_trailing.txt");
            out << body.str() << "\n\n"; // extra blank lines
        }
        orbslam3r::ORBVocabulary trailing;
        check(trailing.loadFromTextFile("/tmp/orbslam3r_voc_trailing.txt") && trailing.size() == voc.size(),
              "trailing blank lines add no node", std::to_string(trailing.size()) + " words");
    }

    // ---------------------------------------------------------------------------
    // dbow2_ext: Boost archives, without the intrusive members ORB-SLAM3 added.
    void TestSerialization()
    {
        std::puts("dbow2_ext :: serialization");

        DBoW2::BowVector bow;
        bow.addWeight(3, 0.25);
        bow.addWeight(9, 0.75);
        DBoW2::FeatureVector fv;
        fv.addFeature(2, 10);
        fv.addFeature(2, 11);
        fv.addFeature(5, 12);

        std::stringstream buf;
        {
            boost::archive::text_oarchive oa(buf);
            oa << bow << fv;
        }
        check(!buf.str().empty(), "non-intrusive serialize() is found", std::to_string(buf.str().size()) + " bytes");

        DBoW2::BowVector bow2;
        DBoW2::FeatureVector fv2;
        {
            boost::archive::text_iarchive ia(buf);
            ia >> bow2 >> fv2;
        }
        check(bow2.size() == 2 && std::abs(bow2[9] - 0.75) < 1e-12, "BowVector round trip");
        check(fv2.size() == 2 && fv2[2].size() == 2 && fv2[5].size() == 1, "FeatureVector round trip");
    }

    // ---------------------------------------------------------------------------
    // g2o_ext: the renamed symbols and the solver-ownership helper.
    void TestG2oCompat()
    {
        std::puts("g2o_ext :: compat + solver factory");

        // Compiles only if the aliases resolve to real upstream types.
        static_assert(std::is_same_v<g2o::VertexSBAPointXYZ, g2o::VertexPointXYZ>);
        static_assert(std::is_same_v<g2o::Vector7d, g2o::Vector7>);
        check(true, "VertexSBAPointXYZ / Vector7d aliases");

        // A two-view bundle adjustment written the way ORB-SLAM3 writes one: a fixed
        // reference pose, a free pose, and points observed by both.
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3>());
        optimizer.setVerbose(false);
        check(optimizer.solver() != nullptr, "MakeLevenberg<BlockSolver_6_3>");

        const double fx = 500, fy = 500, cx = 320, cy = 240;
        const Eigen::Matrix3d R1 = Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Vector3d t1(0.20, -0.03, 0.01);

        auto* pose0 = new g2o::VertexSE3Expmap();
        pose0->setId(0);
        pose0->setEstimate(g2o::SE3Quat());
        pose0->setFixed(true);
        optimizer.addVertex(pose0);

        auto* pose1 = new g2o::VertexSE3Expmap();
        pose1->setId(1);
        pose1->setEstimate(g2o::SE3Quat()); // deliberately wrong: identity
        optimizer.addVertex(pose1);

        const std::vector<Eigen::Vector3d> points = {
            {0.4, 0.2, 4.0},   {-0.6, 0.3, 5.0}, {0.2, -0.5, 3.5}, {1.0, 0.7, 6.0},
            {-0.9, -0.2, 4.5}, {0.1, 0.9, 5.5},  {0.7, -0.8, 4.2}, {-0.4, 0.6, 3.8},
        };
        int id = 2;
        int point_vertices = 0;
        for(const auto &Xw : points)
        {
            auto* p = new g2o::VertexSBAPointXYZ(); // the alias under test
            p->setId(id++);
            p->setEstimate(Xw);
            p->setFixed(true);
            p->setMarginalized(true);
            if(optimizer.addVertex(p))
                ++point_vertices;

            for(int cam = 0; cam < 2; ++cam)
            {
                const Eigen::Vector3d Xc = cam == 0 ? Xw : Eigen::Vector3d(R1 * Xw + t1);
                auto* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, p);
                e->setVertex(1, cam == 0 ? static_cast<g2o::OptimizableGraph::Vertex*>(pose0)
                                         : static_cast<g2o::OptimizableGraph::Vertex*>(pose1));
                e->setMeasurement(Eigen::Vector2d(fx * Xc.x() / Xc.z() + cx, fy * Xc.y() / Xc.z() + cy));
                e->setInformation(Eigen::Matrix2d::Identity());
                e->fx = fx;
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;
                optimizer.addEdge(e);
            }
        }
        check(point_vertices == static_cast<int>(points.size()), "VertexSBAPointXYZ accepted by the graph",
              std::to_string(point_vertices) + " points");

        optimizer.initializeOptimization();
        optimizer.optimize(30);
        optimizer.computeActiveErrors();

        const g2o::SE3Quat est = pose1->estimate();
        const double rot_err = (est.rotation().toRotationMatrix() - R1).norm();
        const double trans_err = (est.translation() - t1).norm();
        check(rot_err < 1e-4 && trans_err < 1e-4, "two-view BA recovers pose 1",
              "|dR|=" + std::to_string(rot_err) + " |dt|=" + std::to_string(trans_err));

        // The other factory entry points must instantiate too.
        g2o::SparseOptimizer gn;
        gn.setAlgorithm(
            orbslam3r::g2o_ext::MakeGaussNewton<g2o::BlockSolverX, orbslam3r::g2o_ext::LinearSolver::kDense>());
        check(gn.solver() != nullptr, "MakeGaussNewton<BlockSolverX, kDense>");
    }

    // ---------------------------------------------------------------------------
    void TestRealVocabulary(const std::string &path)
    {
        std::puts("dbow2_ext :: real ORBvoc.txt");
        orbslam3r::ORBVocabulary voc;
        const bool ok = voc.loadFromTextFile(path);
        check(ok, "loads the production vocabulary", path);
        if(!ok)
            return;
        check(voc.size() > 100000, "word count", std::to_string(voc.size()) + " words");
        check(voc.getBranchingFactor() == 10 && voc.getDepthLevels() == 6, "tree shape k=10 L=6",
              "k=" + std::to_string(voc.getBranchingFactor()) + " L=" + std::to_string(voc.getDepthLevels()));

        const std::vector<cv::Mat> query = RandomDescriptors(500, 4242);
        DBoW2::BowVector bow;
        DBoW2::FeatureVector fv;
        voc.transform(query, bow, fv, 4);
        check(!bow.empty() && !fv.empty(), "transform() produces a BoW vector",
              std::to_string(bow.size()) + " words, " + std::to_string(fv.size()) + " nodes");
        check(std::abs(voc.score(bow, bow) - 1.0) < 1e-9, "self-score is 1.0");
    }

    // The pose-only problem Tracking solves on every frame: one SE3 vertex,
    // unary stereo edges with a Huber kernel. Returns the iterations optimize(10)
    // ran and leaves the estimate in `pose`.
    int SolvePose(g2o::OptimizationAlgorithmLevenberg* algorithm, g2o::SE3Quat &pose)
    {
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(algorithm);

        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setEstimate(g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitY())),
                                    Eigen::Vector3d(0.05, -0.02, 0.10)));
        v->setId(0);
        optimizer.addVertex(v);

        const double fx = 718.856, fy = 718.856, cx = 607.19, cy = 185.22, bf = 386.14;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> ux(-15, 15), uy(-3, 3), uz(5, 40), noise(-1, 1);
        for(int i = 0; i < 200; i++)
        {
            const Eigen::Vector3d Xw(ux(rng), uy(rng), uz(rng));
            const double u = fx * Xw[0] / Xw[2] + cx + noise(rng), w = fy * Xw[1] / Xw[2] + cy + noise(rng);
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            e->setVertex(0, v);
            e->setMeasurement(Eigen::Vector3d(u, w, u - bf / Xw[2] + noise(rng)));
            e->setInformation(Eigen::Matrix3d::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(std::sqrt(7.815));
            e->setRobustKernel(rk);
            e->fx = fx, e->fy = fy, e->cx = cx, e->cy = cy, e->bf = bf, e->Xw = Xw;
            optimizer.addEdge(e);
        }

        optimizer.initializeOptimization(0);
        const int nIterations = optimizer.optimize(10);
        pose = v->estimate();
        return nIterations;
    }

    void TestLevenbergStopOnStall()
    {
        std::puts("\n-- g2o_ext: Levenberg-Marquardt stops once it has stalled --");
        using BlockSolver = g2o::BlockSolver_6_3;
        using Dense = g2o::LinearSolverDense<BlockSolver::PoseMatrixType>;

        g2o::SE3Quat plain, stalled;
        const int nPlain = SolvePose(
            new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BlockSolver>(std::make_unique<Dense>())), plain);
        const int nStalled = SolvePose(
            orbslam3r::g2o_ext::MakeLevenberg<BlockSolver, orbslam3r::g2o_ext::LinearSolver::kDense>(), stalled);

        check(nPlain == 10, "upstream runs every iteration asked for", std::to_string(nPlain) + " of 10");
        check(nStalled >= 4 && nStalled < nPlain, "MakeLevenberg stops after three stalled",
              std::to_string(nStalled) + " of 10");
        const double dDiff = (plain.toVector() - stalled.toVector()).norm();
        check(dDiff < 1e-6, "and reaches the same pose", "difference " + std::to_string(dDiff));
    }

} // namespace

int main(int argc, char** argv)
{
    std::puts("== vendor_ext wrapper tests ==\n");
    TestTextVocabulary();
    TestSerialization();
    TestG2oCompat();
    TestLevenbergStopOnStall();
    if(argc > 1)
        TestRealVocabulary(argv[1]);
    else
        std::puts("\n(pass a path to ORBvoc.txt to also test the real vocabulary)");

    std::printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES: see [FAIL] lines above");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
