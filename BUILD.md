# Dependencies
The following packages are required to build the project:
* pybind11
* torch
* eigen and autodiff (installed automatically with FetchContent)

# Build
Make sure to set Torch_DIR in CMakeLists.txt, then use the following commands to build the project:
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . --config Release