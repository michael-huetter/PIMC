#include "Potential.hpp"
#include <iostream>

PotentialMatrix::ComputeFunction PotentialMatrix::computeFunction_ = 
    [](const Eigen::RowVectorXd& position, std::size_t dim) {
        return potFunctionT<double>(position);
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

Eigen::RowVectorXd PotentialMatrix::gradAutoDiff(const Eigen::RowVectorXd& position, 
                                                 std::size_t i, 
                                                 std::size_t j) const
{
    dual x = position[0];
    dual y = position[1];
    dual z = position[2];

    constexpr auto uFuncs = getUFunctions<dual>();
    auto computeFunc = findComputeFunction<dual>(i, j, uFuncs);

    std::size_t dim = position.size();
    Eigen::RowVectorXd grad(dim);
    
    grad(0) = derivative(computeFunc, wrt(x), at(x, y, z));
    grad(1) = derivative(computeFunc, wrt(y), at(x, y, z));
    grad(2) = derivative(computeFunc, wrt(z), at(x, y, z));

    return grad;
}