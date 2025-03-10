# Dependencies
The following packages are required to build the project:
* pybind11
* torch
* eigen and autodiff (installed automatically with FetchContent)

# Build
Make sure to set `Torch_DIR` in CMakeLists.txt, then use the following commands to build the project:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . --config Release


# CUDA on LEO5
Download LibTorch/CDUA11.8 (Pre-cxx11), to build use:
    
    conda create -n pimcV0 python=3.11
    conda install pip pybind11
    pip install numpy matplotlib
    module load cuda/11.8.0-gcc-11.3.0-tqwsjwi
    module load gcc/11.3.0-gcc-8.5.0-rwipohd
    export PATH=/usr/site/hpc/spack/v0.19-leo5-20230116/opt/spack/linux-rocky8-icelake/gcc-11.3.0/cuda-11.8.0-tqwsjwi4bm662fzv574xhhfa2tllyxbg/bin:$PATH
    export LD_LIBRARY_PATH=/usr/site/hpc/spack/v0.19-leo5-20230116/opt/spack/linux-rocky8-icelake/gcc-11.3.0/cuda-11.8.0-tqwsjwi4bm662fzv574xhhfa2tllyxbg/lib64:$LD_LIBRARY_PATH
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . --config Release
    export LD_LIBRARY_PATH=/usr/site/hpc/spack/v0.19-leo5-20230116/opt/spack/linux-rocky8-icelake/gcc-8.5.0/gcc-11.3.0-rwipohddc5mr565rtbaulkeks5xnijlc/lib64:$LD_LIBRARY_PATH