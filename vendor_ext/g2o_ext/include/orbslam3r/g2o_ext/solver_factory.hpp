// =============================================================================
// orbslam3r/g2o_ext/solver_factory.hpp
//
// One place that knows how modern g2o wants its solvers constructed.
//
// THE PROBLEM THIS SOLVES
//   ORB-SLAM3 v1.0 builds optimizers the way g2o worked in 2016 -- raw owning
//   pointers threaded through three constructors:
//
//     auto* linear = new g2o::LinearSolverEigen<BlockSolver_6_3::PoseMatrixType>();
//     auto* block  = new g2o::BlockSolver_6_3(linear);
//     auto* algo   = new g2o::OptimizationAlgorithmLevenberg(block);
//     optimizer.setAlgorithm(algo);
//
//   Current g2o takes std::unique_ptr at both levels, so every one of those
//   sites is a compile error.  Optimizer.cc alone has a dozen of them, each
//   spelled slightly differently.
//
//   Rather than fixing them one by one, route them all through here.  When g2o
//   changes its ownership convention again, this header is the only thing that
//   has to move.
//
// USAGE
//   g2o::SparseOptimizer optimizer;
//   optimizer.setAlgorithm(orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolver_6_3>());
//
//   // dense linear solver instead of the sparse Eigen one:
//   optimizer.setAlgorithm(
//       orbslam3r::g2o_ext::MakeLevenberg<g2o::BlockSolverX,
//                                         orbslam3r::g2o_ext::LinearSolver::kDense>());
// =============================================================================
#pragma once

#include <memory>
#include <utility>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

namespace orbslam3r::g2o_ext {

enum class LinearSolver {
  kEigen,  // sparse Cholesky; what ORB-SLAM3 uses nearly everywhere
  kDense,  // dense Cholesky; used for the small inertial-only problems
};

namespace detail {

template <typename BlockSolverT, LinearSolver kind>
std::unique_ptr<g2o::Solver> MakeBlockSolver() {
  using PoseMatrix = typename BlockSolverT::PoseMatrixType;
  if constexpr (kind == LinearSolver::kEigen) {
    return std::make_unique<BlockSolverT>(
        std::make_unique<g2o::LinearSolverEigen<PoseMatrix>>());
  } else {
    return std::make_unique<BlockSolverT>(
        std::make_unique<g2o::LinearSolverDense<PoseMatrix>>());
  }
}

}  // namespace detail

// Both return a raw pointer because that is what SparseOptimizer::setAlgorithm
// takes: the optimizer assumes ownership of the algorithm it is given.
template <typename BlockSolverT, LinearSolver kind = LinearSolver::kEigen>
g2o::OptimizationAlgorithmLevenberg* MakeLevenberg() {
  return new g2o::OptimizationAlgorithmLevenberg(
      detail::MakeBlockSolver<BlockSolverT, kind>());
}

template <typename BlockSolverT, LinearSolver kind = LinearSolver::kEigen>
g2o::OptimizationAlgorithmGaussNewton* MakeGaussNewton() {
  return new g2o::OptimizationAlgorithmGaussNewton(
      detail::MakeBlockSolver<BlockSolverT, kind>());
}

}  // namespace orbslam3r::g2o_ext
