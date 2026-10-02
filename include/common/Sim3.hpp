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

#ifndef COMMON_SIM3_H
#define COMMON_SIM3_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3
{

    // A similarity of space: x -> s R x + t. Loop Closing's poses are these,
    // because a monocular map drifts in scale as well as in place.
    //
    // The arithmetic is that of g2o's Sim3, expression for expression, which
    // this stands in for outside src/optim_g2o/: the same inputs give the same
    // bits. That includes what its constructors do -- the rotation is made to
    // have w >= 0 and unit norm -- and what the product and the accessors do
    // not.
    struct Sim3
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        Eigen::Quaterniond r;
        Eigen::Vector3d t;
        double s;

    public:
        Sim3()
        {
            r.setIdentity();
            t.fill(0.);
            s = 1.;
        }

        Sim3(const Eigen::Quaterniond &r, const Eigen::Vector3d &t, double s) : r(r), t(t), s(s)
        {
            normalizeRotation();
        }

        Sim3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double s) : r(Eigen::Quaterniond(R)), t(t), s(s)
        {
            normalizeRotation();
        }

        Eigen::Vector3d map(const Eigen::Vector3d &xyz) const { return s * (r * xyz) + t; }

        Sim3 inverse() const { return Sim3(r.conjugate(), r.conjugate() * ((-1 / s) * t), 1 / s); }

        Sim3 operator*(const Sim3 &other) const
        {
            Sim3 ret;
            ret.r = r * other.r;
            ret.t = s * (r * other.t) + t;
            ret.s = s * other.s;
            return ret;
        }

        Sim3 &operator*=(const Sim3 &other)
        {
            Sim3 ret = (*this) * other;
            *this = ret;
            return *this;
        }

        void normalizeRotation()
        {
            if(r.w() < 0)
            {
                r.coeffs() *= -1;
            }
            r.normalize();
        }

        inline const Eigen::Vector3d &translation() const { return t; }

        inline Eigen::Vector3d &translation() { return t; }

        inline const Eigen::Quaterniond &rotation() const { return r; }

        inline Eigen::Quaterniond &rotation() { return r; }

        inline const double &scale() const { return s; }

        inline double &scale() { return s; }
    };

} // namespace ORB_SLAM3

#endif // COMMON_SIM3_H
