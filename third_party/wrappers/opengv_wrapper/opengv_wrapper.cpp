#include "opengv_wrapper.hpp"
#include <cmath>

namespace slam3 { namespace geometry {

Rotation rodrigues2rot(const Eigen::Vector3d &omega)
{
    double theta = omega.norm();
    if(theta < 1e-10)
        return Rotation::Identity();

    Eigen::Vector3d k = omega / theta;
    Eigen::Matrix3d K;
    K <<     0, -k(2),  k(1),
          k(2),     0, -k(0),
         -k(1),  k(0),     0;

    return Rotation::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

Rodrigues rot2rodrigues(const Rotation &R)
{
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

BearingVector pixelToBearing(double u, double v,
                             double fx, double fy, double cx, double cy)
{
    Eigen::Vector3d bearing((u - cx) / fx, (v - cy) / fy, 1.0);
    return bearing.normalized();
}

}} // namespace slam3::geometry
