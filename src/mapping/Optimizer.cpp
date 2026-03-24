/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
* Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
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

#include "Optimizer.hpp"
#include "G2oOptimizer.hpp"

#include <Eigen/Dense>

namespace ORB_SLAM3
{

// Static backend storage
std::unique_ptr<IOptimizer> Optimizer::sBackend_;

void Optimizer::SetBackend(std::unique_ptr<IOptimizer> backend)
{
    sBackend_ = std::move(backend);
}

IOptimizer& Optimizer::Get()
{
    if (!sBackend_)
        sBackend_ = std::make_unique<G2oOptimizer>();
    return *sBackend_;
}

// Marginalize block element (start:end,start:end). Perform Schur complement.
// Marginalized elements are filled with zeros.
// Pure linear algebra — not backend-specific.
Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end-start+1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end+1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}

} //namespace ORB_SLAM3
