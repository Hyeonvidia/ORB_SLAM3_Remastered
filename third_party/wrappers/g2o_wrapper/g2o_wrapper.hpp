#pragma once

// =============================================================================
// g2o Wrapper — Unified include for g2o-orbslam3 fork headers
// =============================================================================

// Core
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/sparse_block_matrix.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/robust_kernel_impl.h"

// Solvers
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/solvers/linear_solver_dense.h"

// Types
#include "g2o/types/types_sba.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/types/types_seven_dof_expmap.h"
#include "g2o/types/sim3.h"
