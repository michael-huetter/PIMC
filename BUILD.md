# Dependencies
The following packages are required to build the project:
* pybind11
* torch
* eigen and autodiff (installed automatically with FetchContent)
Make sure to set Torch_DIR in CMakeLists.txt!

# Build
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . --config Release