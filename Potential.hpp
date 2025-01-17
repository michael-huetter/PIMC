#pragma once

#include <cstddef>
#include <functional>
#include <Eigen/Dense>

class PotentialMatrix {
public:
    using ComputeFunction = std::function<Eigen::MatrixXd(const Eigen::RowVectorXd&, std::size_t)>;
    explicit PotentialMatrix(std::size_t dim);
    Eigen::MatrixXd compute(const Eigen::RowVectorXd& position) const;
    static void setComputeFunction(ComputeFunction f);

private:
    std::size_t dim_; // relates to the number of electronic states in the system, not the spatial dimension
    static ComputeFunction computeFunction_;
};

