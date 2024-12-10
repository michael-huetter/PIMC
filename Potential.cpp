#include "Potential.hpp"

PotentialMatrix::PotentialMatrix(std::size_t dim)
    : dim_(dim)
{
}

Eigen::MatrixXd PotentialMatrix::compute(const Eigen::RowVectorXd& position) const
{
    double x = position[0];
    double y = position[1];
    double z = position[2];

    Eigen::MatrixXd pot_mat = Eigen::MatrixXd::Zero(dim_, dim_);

    pot_mat(0,0) = 0.5 * (x*x + y*y + z*z);
    pot_mat(1,1) = 0.5 * (x*x + y*y + z*z) + 1.0;

    return pot_mat;
}

