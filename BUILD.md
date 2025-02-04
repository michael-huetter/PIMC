# Dependencies
The following packages are required to build the project:
* pybind11
* torch
* eigen and autodiff (installed automatically with FetchContent)

# Build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make