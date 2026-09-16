// Exercises each third-party dependency the SLAM pipeline relies on, so a
// broken image fails here in seconds instead of during a dataset run.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <sophus/se3.hpp>

#include <DBoW2/FORB.h>
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>

#include <pangolin/pangolin.h>

static int failures = 0;

static void check(bool ok, const char* what, const std::string &detail = "")
{
    std::printf("  [%s] %-34s %s\n", ok ? " ok " : "FAIL", what, detail.c_str());
    if(!ok)
        ++failures;
}

// ---------------------------------------------------------------------------
static void test_eigen_opencv()
{
    std::puts("Eigen / OpenCV");
    Eigen::Matrix3d R = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    check(std::abs(R.determinant() - 1.0) < 1e-12, "rotation matrix is orthonormal");

    cv::Mat m = cv::Mat::zeros(4, 4, CV_8U);
    check(m.total() == 16 && m.type() == CV_8U, "cv::Mat allocation", std::string("OpenCV ") + CV_VERSION);
}

// ---------------------------------------------------------------------------
static void test_sophus()
{
    std::puts("Sophus");
    const Eigen::Vector3d omega(0.1, -0.2, 0.3);
    const Eigen::Vector3d t(1.0, 2.0, 3.0);
    Sophus::SE3d T(Sophus::SO3d::exp(omega), t);

    // log(exp(x)) must return the tangent vector it started from.
    const Sophus::SE3d::Tangent xi = T.log();
    const Sophus::SE3d T2 = Sophus::SE3d::exp(xi);
    check((T.matrix() - T2.matrix()).norm() < 1e-10, "SE3 exp/log round-trip");
    check((T.inverse() * T).log().norm() < 1e-12, "SE3 inverse");
}

// ---------------------------------------------------------------------------
static void test_dbow2()
{
    std::puts("DBoW2");
    // Two 32-byte ORB descriptors differing in a single bit.
    cv::Mat a = cv::Mat::zeros(1, 32, CV_8U);
    cv::Mat b = cv::Mat::zeros(1, 32, CV_8U);
    b.at<uint8_t>(0, 5) = 0x08;
    const int d = DBoW2::FORB::distance(a, b);
    check(d == 1, "FORB::distance Hamming", "distance=" + std::to_string(d));

    DBoW2::BowVector bow;
    bow.addWeight(7, 0.5);
    bow.addWeight(7, 0.25);
    check(bow.size() == 1 && std::abs(bow[7] - 0.75) < 1e-12, "BowVector::addWeight");

    DBoW2::FeatureVector fv;
    fv.addFeature(3, 11);
    fv.addFeature(3, 12);
    check(fv.size() == 1 && fv.begin()->second.size() == 2, "FeatureVector::addFeature");
}

// ---------------------------------------------------------------------------
static void test_g2o()
{
    std::puts("g2o");
    // This is the modern ownership convention: the block solver takes a
    // unique_ptr to the linear solver, and the algorithm takes a unique_ptr to
    // the block solver.  ORB-SLAM3 v1.0 still passes raw `new` pointers, which
    // is the single most common compile break when moving to current g2o.
    using BlockSolver = g2o::BlockSolver_6_3;
    using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    auto linear = std::make_unique<LinearSolver>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block)));
    optimizer.setVerbose(false);
    check(optimizer.solver() != nullptr, "SparseOptimizer + Levenberg");

    // A pose-only bundle adjustment, the same shape as ORB-SLAM3's
    // Optimizer::PoseOptimization: one camera pose and the reprojection of
    // several known 3D points as unary edges.
    const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
    const Eigen::Matrix3d R_true = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d t_true(0.10, -0.05, 0.02);

    auto* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat()); // start from identity
    optimizer.addVertex(pose);
    check(optimizer.vertices().size() == 1, "VertexSE3Expmap (types_sba)");

    const std::vector<Eigen::Vector3d> points = {
        {0.5, 0.2, 4.0}, {-0.7, 0.4, 5.0}, {0.1, -0.6, 3.5}, {1.2, 0.9, 6.0}, {-1.1, -0.3, 4.5}, {0.3, 0.8, 5.5},
    };
    size_t edges = 0;
    for(const auto &Xw : points)
    {
        // The observation is the exact projection under the true pose, so the
        // optimum is the true pose and the converged error must be ~0.
        const Eigen::Vector3d Xc = R_true * Xw + t_true;
        auto* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
        e->setVertex(0, pose);
        e->setMeasurement(Eigen::Vector2d(fx * Xc.x() / Xc.z() + cx, fy * Xc.y() / Xc.z() + cy));
        e->setInformation(Eigen::Matrix2d::Identity());
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        e->Xw = Xw;
        auto* kernel = new g2o::RobustKernelHuber();
        kernel->setDelta(std::sqrt(5.991));
        e->setRobustKernel(kernel);
        if(optimizer.addEdge(e))
            ++edges;
    }
    check(edges == points.size(), "EdgeSE3ProjectXYZOnlyPose", std::to_string(edges) + " edges");

    optimizer.initializeOptimization();
    const int iters = optimizer.optimize(20);
    check(iters > 0, "optimize() runs", "iterations=" + std::to_string(iters));

    optimizer.computeActiveErrors();
    const double chi2 = optimizer.activeChi2();
    check(chi2 < 1e-6, "pose-only BA converges", "chi2=" + std::to_string(chi2));

    const g2o::SE3Quat est = pose->estimate();
    const double rot_err = (est.rotation().toRotationMatrix() - R_true).norm();
    const double trans_err = (est.translation() - t_true).norm();
    check(rot_err < 1e-5 && trans_err < 1e-5, "recovers the true pose",
          "|dR|=" + std::to_string(rot_err) + " |dt|=" + std::to_string(trans_err));

    // Sim3 lives in a separate g2o library; instantiate one to prove it links.
    auto sim3 = std::make_unique<g2o::VertexSim3Expmap>();
    sim3->setEstimate(g2o::Sim3());
    check(std::abs(sim3->estimate().scale() - 1.0) < 1e-12, "VertexSim3Expmap (types_sim3)");
}

// ---------------------------------------------------------------------------
static void test_pangolin(bool open_window)
{
    std::puts("Pangolin");
    check(true, "headers + link", std::string("Pangolin ") + ORBSLAM3R_PANGOLIN_VERSION);
    if(!open_window)
    {
        std::puts("  [skip] window creation          (pass --gl to try it)");
        return;
    }
    try
    {
        pangolin::CreateWindowAndBind("orbslam3r-smoke", 320, 240);
        const std::string renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        pangolin::DestroyWindow("orbslam3r-smoke");
        check(!renderer.empty(), "GL context on current DISPLAY", renderer);
    }
    catch(const std::exception &e)
    {
        check(false, "GL context on current DISPLAY", e.what());
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    bool open_window = false;
    for(int i = 1; i < argc; ++i)
        if(std::strcmp(argv[i], "--gl") == 0)
            open_window = true;

    std::puts("== ORB_SLAM3_Remastered dependency smoke test ==\n");
    test_eigen_opencv();
    test_sophus();
    test_dbow2();
    test_g2o();
    test_pangolin(open_window);

    std::printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES: see [FAIL] lines above");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
