// =============================================================================
// g2o's sparse Eigen solver with the factorisation ORB-SLAM3 was written for.
//
// This one is not a change ORB-SLAM3 made. It is a change upstream made after
// ORB-SLAM3 froze its copy, so tools/upstream_delta.py, which reports what
// ORB-SLAM3 edited, cannot see it. The fork's LinearSolverEigen factorises with
// Eigen::SimplicialLDLT; upstream switched to SimplicialLLT in 2020 (g2o
// f82ee904, "use LLT instead of LDLT Cholesky decomposition").
//
// The two fail differently. LDLT gives up only on a pivot that is exactly zero;
// LLT on any pivot that is not positive. A visual-inertial bundle adjustment
// has gauge freedom -- yaw, and position -- that nothing fixes, and ORB-SLAM3
// starts Full Inertial BA at lambda 1e-5, so its Hessian is positive definite
// only on paper. The 765 x 765 one upstream dumped from a EuRoC run factors
// with a single pivot of -0.0185 among 764 positive ones: v1.0 solved it and
// moved on, upstream's LLT returns false, writes 1.3 MB of "debug.txt" from the
// Local Mapping thread, and Levenberg-Marquardt inflates lambda and tries
// again. Every inertial run logged between 1 and 54 of those; no other
// configuration logged any. docs/PORTING.md used to say v1.0 failed the same
// way and only the logging was new. It did not fail.
//
// The class is upstream's LinearSolverEigen with the decomposition swapped --
// the decomposition is a member there, not a virtual, so it cannot be
// subclassed in. What is left out is the marginal-covariance path
// (solveBlocks / solvePattern): MarginalCovarianceCholesky wants the factor of
// an LLT, nothing in the system asks for marginals, and refusing is better than
// answering from the wrong factor.
//
// The fill-reducing ordering is left at upstream's default (AMD on the block
// structure). The fork's was AMD on the scalar structure. That changes fill-in
// and rounding order, not the solution.
// =============================================================================
#pragma once

#include <cassert>
#include <functional>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <g2o/core/batch_stats.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/marginal_covariance_cholesky.h>
#include <g2o/stuff/logger.h>
#include <g2o/stuff/timeutil.h>

namespace orbslam3r::g2o_ext
{

    template<typename MatrixType>
    class LinearSolverEigenLDLT : public g2o::LinearSolverCCS<MatrixType>
    {
    public:
        using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
        using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
        using DecompositionBase = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>;

        // SimplicialLDLT with the ordering handed in, as the fork has it.
        class Decomposition : public DecompositionBase
        {
        public:
            void analyzePatternWithPermutation(SparseMatrix &a, const PermutationMatrix &permutation)
            {
                this->m_Pinv = permutation;
                this->m_P = permutation.inverse();
                const int size = static_cast<int>(a.cols());
                SparseMatrix ap(size, size);
                ap.template selfadjointView<Eigen::Upper>() = a.template selfadjointView<DecompositionBase::UpLo>()
                                                                  .twistedBy(this->m_P);
                this->analyzePattern_preordered(ap, true); // true: LDLT
            }
        };

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        bool init() override
        {
            mbInit = true;
            return true;
        }

        bool solve(const g2o::SparseBlockMatrix<MatrixType> &A, double* x, double* b) override
        {
            if(mbInit)
                mSparseMatrix.resize(A.rows(), A.cols());
            FillSparseMatrix(A, !mbInit);
            if(mbInit)
                ComputeSymbolicDecomposition(A);
            mbInit = false;

            const double t = g2o::get_monotonic_time();
            mDecomposition.factorize(mSparseMatrix);
            if(mDecomposition.info() != Eigen::Success) // a pivot that is exactly zero
            {
                if(this->writeDebug())
                {
                    G2O_ERROR("LDLT failure, writing debug.txt (Hessian loadable by Octave)");
                    A.writeOctave("debug.txt");
                }
                return false;
            }

            g2o::VectorX::MapType xx(x, mSparseMatrix.cols());
            g2o::VectorX::ConstMapType bb(b, mSparseMatrix.cols());
            xx = mDecomposition.solve(bb);

            if(g2o::G2OBatchStatistics* stats = g2o::G2OBatchStatistics::globalStats())
            {
                stats->timeNumericDecomposition = g2o::get_monotonic_time() - t;
                stats->choleskyNNZ = mDecomposition.matrixL().nestedExpression().nonZeros() + mSparseMatrix.cols();
            }
            return true;
        }

    protected:
        bool solveBlocks_impl(const g2o::SparseBlockMatrix<MatrixType> &,
                              std::function<void(g2o::MarginalCovarianceCholesky &)>) override
        {
            G2O_ERROR("LinearSolverEigenLDLT does not compute marginal covariances");
            return false;
        }

    private:
        void FillSparseMatrix(const g2o::SparseBlockMatrix<MatrixType> &A, bool bOnlyValues)
        {
            if(bOnlyValues)
            {
                this->_ccsMatrix->fillCCS(mSparseMatrix.valuePtr(), true);
                return;
            }
            this->initMatrixStructure(A);
            mSparseMatrix.resizeNonZeros(A.nonZeros());
            const int nz = this->_ccsMatrix->fillCCS(mSparseMatrix.outerIndexPtr(), mSparseMatrix.innerIndexPtr(),
                                                     mSparseMatrix.valuePtr(), true);
            (void)nz;
            assert(nz <= static_cast<int>(mSparseMatrix.data().size()));
        }

        void ComputeSymbolicDecomposition(const g2o::SparseBlockMatrix<MatrixType> &A)
        {
            const double t = g2o::get_monotonic_time();
            if(!this->blockOrdering())
            {
                mDecomposition.analyzePattern(mSparseMatrix);
            }
            else
            {
                assert(A.rows() == A.cols() && "Matrix A is not square");
                PermutationMatrix blockP;
                {
                    SparseMatrix auxBlockMatrix(A.blockCols().size(), A.blockCols().size());
                    auxBlockMatrix.resizeNonZeros(A.nonZeroBlocks());
                    A.fillBlockStructure(auxBlockMatrix.outerIndexPtr(), auxBlockMatrix.innerIndexPtr());
                    Eigen::AMDOrdering<SparseMatrix::StorageIndex> ordering;
                    ordering(auxBlockMatrix, blockP);
                }
                PermutationMatrix scalarP(A.rows());
                this->blockToScalarPermutation(A, blockP.indices(), scalarP.indices());
                mDecomposition.analyzePatternWithPermutation(mSparseMatrix, scalarP);
            }
            if(g2o::G2OBatchStatistics* stats = g2o::G2OBatchStatistics::globalStats())
                stats->timeSymbolicDecomposition = g2o::get_monotonic_time() - t;
        }

        bool mbInit = true;
        SparseMatrix mSparseMatrix;
        Decomposition mDecomposition;
    };

} // namespace orbslam3r::g2o_ext
