#pragma once

#include <cstddef>
#include <Eigen/Dense>

class PotentialMatrix {
public:

    explicit PotentialMatrix(std::size_t dim);
    Eigen::MatrixXd compute(const Eigen::RowVectorXd& position) const;

private:
    std::size_t dim_;
};

