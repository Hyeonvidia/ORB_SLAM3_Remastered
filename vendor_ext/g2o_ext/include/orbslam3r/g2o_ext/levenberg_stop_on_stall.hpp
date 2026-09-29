// =============================================================================
// Levenberg-Marquardt that stops once it has stalled, as ORB-SLAM3's g2o does.
//
// ORB-SLAM3's fork of g2o carries one change to the optimisation algorithm
// itself (docs/modifications/g2o/core__optimization_algorithm_levenberg.cpp.diff,
// "Stop criterium (Raul)"): an iteration that improves the robust chi2 by less
// than a thousandth counts as bad, and after three bad iterations in a row
// solve() returns Terminate, which ends SparseOptimizer::optimize(). Upstream
// has no such rule and always runs the iterations it was asked for.
//
// Nothing else in the system asks for fewer: Optimizer::PoseOptimization calls
// optimize(10) four times on every frame, twice per frame, and converges long
// before the tenth iteration. Without the rule it kept iterating on a solved
// problem -- 1.8 times as many linearisations and solves for the same answer to
// nine digits -- which cost tracking 0.4 ms a frame in monocular and 1.1 ms in
// stereo against v1.0. With it the iteration counts match the fork's exactly.
//
// Why a subclass rather than g2o's own SparseOptimizerTerminateAction: that
// action stops on the first small gain rather than the third, measures the gain
// against the new chi2 rather than the old, and works by taking over the
// optimizer's force-stop flag -- which Local Mapping already uses to abort a
// bundle adjustment. Returning Terminate from solve() is what the fork does,
// and leaves the flag alone.
//
// The chi2 at the start of an iteration is the accepted chi2 of the one before.
// Only the first has to be computed, which costs one pass over the active
// edges per optimize(); upstream's solve() does not expose its own.
//
// Two things a caller may set, which the fork had no means to: after how many
// bad iterations in a row to stop, and how large the damping starts. g2o
// starts it at 1e-5 of the largest diagonal entry of the Hessian, which in a
// bundle adjustment is a keyframe's rotation, thousands of times larger than
// a point's: measured on local bundle adjustments, that damping is as large
// as the points' own entries and takes six or seven iterations, shrinking by
// three each time, to get out of their way.
// =============================================================================
#pragma once

#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>

namespace orbslam3r::g2o_ext
{

    class LevenbergStopOnStall : public g2o::OptimizationAlgorithmLevenberg
    {
    public:
        using g2o::OptimizationAlgorithmLevenberg::OptimizationAlgorithmLevenberg;

        SolverResult solve(int iteration, bool online = false) override
        {
            if(iteration == 0)
            {
                mnBad = 0;
                _optimizer->computeActiveErrors();
                mChi = _optimizer->activeRobustChi2();
            }
            const double iniChi = mChi;

            const SolverResult result = g2o::OptimizationAlgorithmLevenberg::solve(iteration, online);
            if(result != OK)
                return result;

            // After OK the active errors are those of the accepted step. If the
            // last trial was rejected because a stop was requested they are that
            // trial's, which is worse than iniChi and counts as bad -- as it does
            // in the fork, where the chi2 is then unchanged.
            mChi = _optimizer->activeRobustChi2();
            if((iniChi - mChi) * 1e3 < iniChi)
                ++mnBad;
            else
                mnBad = 0;

            return mnBad >= mnStall ? Terminate : OK;
        }

        // Stop after n iterations in a row that each improved the robust chi2
        // by less than a thousandth. Three unless set.
        void setStallIterations(int n) { mnStall = n; }

        // The damping of the first iteration, as a fraction of the largest
        // diagonal entry of the Hessian. g2o's 1e-5 unless set; a user lambda
        // (setUserLambdaInit) takes precedence, as it does in g2o.
        void setInitialDamping(double tau) { _tau = tau; }

    private:
        int mnBad = 0;
        int mnStall = 3;
        double mChi = 0.0;
    };

} // namespace orbslam3r::g2o_ext
