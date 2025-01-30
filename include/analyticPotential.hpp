/*                                                                                           
▗▄▄▄ ▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖    ▗▄▄▖  ▗▄▖▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖ ▗▄▖ ▗▖       ▗▖ ▗▖▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖
▐▌  █▐▌   ▐▌     █  ▐▛▚▖▐▌▐▌       ▐▌ ▐▌▐▌ ▐▌ █  ▐▌   ▐▛▚▖▐▌  █    █  ▐▌ ▐▌▐▌       ▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌   
▐▌  █▐▛▀▀▘▐▛▀▀▘  █  ▐▌ ▝▜▌▐▛▀▀▘    ▐▛▀▘ ▐▌ ▐▌ █  ▐▛▀▀▘▐▌ ▝▜▌  █    █  ▐▛▀▜▌▐▌       ▐▛▀▜▌▐▛▀▀▘▐▛▀▚▖▐▛▀▀▘
▐▙▄▄▀▐▙▄▄▖▐▌   ▗▄█▄▖▐▌  ▐▌▐▙▄▄▖    ▐▌   ▝▚▄▞▘ █  ▐▙▄▄▖▐▌  ▐▌  █  ▗▄█▄▖▐▌ ▐▌▐▙▄▄▖    ▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▐▙▄▄▖                                                                                                                                                                                                                                                                                                                   
--------------------------------------------------------------------------------------------------------
Costume analytic potential can be defined here in the form of U<i, j, T> 
where i and j are the row and column indices of the potential matrix.
Note:
 - the functions need to be templated on the type T
 - recompile the code after making changes to this file to apply the changes
 - if the virial estimator is used, the gradient is automatically computed using automatic differentiation
--------------------------------------------------------------------------------------------------------
*/

template <typename T>
struct U<0, 0, T> {
    static T compute(T x, T y, T z) {
        return 0.5 * (x * x + y * y + z * z);
    }
};

template <typename T>
struct U<1, 1, T> {
    static T compute(T x, T y, T z) {
        return 0.5 * (x * x + y * y + z * z) + T(1.0);
    }
};

template <typename T>
constexpr auto getUFunctions() {
    return std::make_tuple(
        FunctionEntry<T, 0, 0>(),
        FunctionEntry<T, 1, 1>()
    );
}