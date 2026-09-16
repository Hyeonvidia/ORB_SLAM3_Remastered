// =============================================================================
// orbslam3r/g2o_ext/compat.hpp
//
// Name compatibility between the g2o ORB-SLAM3 forked (~2016) and the pinned
// upstream release (20241228_git).
//
// WHAT ORB-SLAM3 ACTUALLY CHANGED IN g2o
//   Measured with tools/upstream_delta.py against the closest ancestor blob in
//   upstream history: 1105 lines across 63 files.  Essentially all of it is
//   packaging -- `#include "g2o/core/x.h"` rewritten to `#include "../core/x.h"`
//   so the stripped tree builds standalone -- plus the drift of an eight-year-old
//   fork.  The one file with no upstream counterpart at all, types/se3mat.h,
//   is dead code: nothing in ORB-SLAM3's src/ or include/ references SE3mat, and
//   its ExpSO3/LogSO3 duplicate the ones in ORB-SLAM3's own G2oTypes.h.
//
//   So no algorithm needs porting.  Of the 26 distinct g2o symbols ORB-SLAM3
//   names, upstream 20241228 already provides 24 under the same name.  The two
//   below were renamed upstream, and that is the entire compatibility surface.
//
// HOW TO USE
//   Include this instead of reaching for g2o headers directly.  Code may keep
//   writing g2o::VertexSBAPointXYZ and g2o::Vector7d; both resolve to the
//   upstream types with no conversion and no wrapper object in between.
// =============================================================================
#pragma once

#include <g2o/core/eigen_types.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

namespace g2o {

// Upstream folded the SBA-specific point vertex into the shared slam3d types
// and dropped the "SBA" prefix.  Same base class, same estimate type:
//   BaseVertex<3, Vector3>
// ORB-SLAM3 names this 66 times across Optimizer.cc alone.
using VertexSBAPointXYZ = VertexPointXYZ;

// Upstream's fixed-size vector aliases are Vector2/Vector3/.../Vector7 (built
// on VectorN<N>); ORB-SLAM3 inherited the older `...d` spelling from se3quat.h,
// where it was a plain Matrix<double,7,1>.  Identical type, different name.
using Vector7d = Vector7;

}  // namespace g2o
