#include "Potential.hpp"
#include <iostream>

PotentialMatrix::ComputeFunction PotentialMatrix::computeFunction_ = 
    [](const Eigen::RowVectorXd& position, std::size_t dim) {
        double x = position[0];
        double y = position[1];
        double z = position[2];

        // Define the potential matrix here (recompile if you want to change the potential)
        Eigen::MatrixXd pot_mat = Eigen::MatrixXd::Zero(dim, dim);

        pot_mat(0,0) = 0.5 * (x*x + y*y + z*z);
        pot_mat(1,1) = 0.5 * (x*x + y*y + z*z) + 1.0;

        return pot_mat;
    };

PotentialMatrix::PotentialMatrix(std::size_t dim)
    : dim_(dim)
{
}

Eigen::MatrixXd PotentialMatrix::compute(const Eigen::RowVectorXd& position) const
{
    return computeFunction_(position, dim_);
}

void PotentialMatrix::setComputeFunction(ComputeFunction f)
{
    computeFunction_ = f;
}