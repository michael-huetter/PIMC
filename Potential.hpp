#pragma once

#include "analyticPotential.hpp"
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>

using namespace autodiff;

// Automatic construction of potential matrix 
template <int i, int j, typename T>
struct U;

template <typename T, int i, int j>
struct FunctionEntry {
    static constexpr int row = i;
    static constexpr int col = j;
    using FuncType = T(*)(T, T, T);
    static FuncType func() {
        return &U<i, j, T>::compute;
    }
};
template <typename T, typename Tuple, std::size_t... Is>
void setMatrixElements(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat,
                       const Tuple& funcs,
                       const Eigen::Matrix<T, 1, Eigen::Dynamic>& position,
                       std::index_sequence<Is...>) {
    (void)std::initializer_list<int>{
        (
            mat(std::tuple_element_t<Is, Tuple>::row,
                std::tuple_element_t<Is, Tuple>::col) =
                std::tuple_element_t<Is, Tuple>::func()(position[0], position[1], position[2]),
            0
        )...
    };
}
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
potFunctionT(const Eigen::Matrix<T, 1, Eigen::Dynamic>& position) {
    constexpr int numFunctions = std::tuple_size<decltype(getUFunctions<T>())>::value;
    auto uFuncs = getUFunctions<T>();
    
    int maxIndex = 0;
    std::apply([&maxIndex](auto&&... entry) {
        ((maxIndex = std::max(maxIndex, entry.row), maxIndex = std::max(maxIndex, entry.col)), ...);
    }, uFuncs);
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pot_mat(maxIndex + 1, maxIndex + 1);
    pot_mat.setZero();

    setMatrixElements<T>(pot_mat, uFuncs, position, std::make_index_sequence<numFunctions>{});

    return pot_mat;
}

template<typename T, typename Tuple>
std::function<T(T, T, T)> findComputeFunction(std::size_t i, std::size_t j, const Tuple& funcs)
{
    std::function<T(T, T, T)> result = nullptr;
    std::apply([&](auto&&... entry) {
        (
            [&] {
                if (entry.row == i && entry.col == j) {
                    result = entry.func();  
                }
            }(),
            ...
        );
    }, funcs);
    return result;
}

class PotentialMatrix {
public:
    using ComputeFunction = std::function<Eigen::MatrixXd(const Eigen::RowVectorXd&, std::size_t)>;
    explicit PotentialMatrix(std::size_t dim);
    Eigen::MatrixXd compute(const Eigen::RowVectorXd& position) const;
    static void setComputeFunction(ComputeFunction f);
    Eigen::RowVectorXd gradAutoDiff(const Eigen::RowVectorXd& position,
                                    std::size_t i,
                                    std::size_t j) const;

private:
    std::size_t dim_; // relates to the number of electronic states in the system, not the spatial dimension
    static ComputeFunction computeFunction_;
};

