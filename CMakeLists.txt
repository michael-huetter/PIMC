cmake_minimum_required(VERSION 3.14)
project(PIMC VERSION 1.2 LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
set(BUILD_TESTING OFF CACHE BOOL "Disable all tests" FORCE)
set(EIGEN_BUILD_TESTS OFF CACHE BOOL "Disable Eigen tests" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/autodiff/autodiff.git
  GIT_TAG        v1.1.2
)
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "Disable autodiff tests" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "Disable autodiff examples" FORCE)
FetchContent_MakeAvailable(eigen autodiff)

find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(
    PIMC
    src/binding.cpp
    src/mcmc.cpp
    src/Beads.cpp
    src/Energy.cpp
    src/Potential.cpp
)

target_include_directories(PIMC
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
    ${autodiff_SOURCE_DIR}
)

target_compile_options(PIMC PRIVATE -O3 -Wall -Wextra)
set_target_properties(PIMC PROPERTIES
        OUTPUT_NAME "PIMC"
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
install(TARGETS PIMC LIBRARY DESTINATION .)