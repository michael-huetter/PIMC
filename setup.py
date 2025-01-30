from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "PIMC",
        ["src/binding.cpp", "src/mcmc.cpp", "src/Beads.cpp", "src/Energy.cpp", "src/Potential.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3",
            "/opt/homebrew/Cellar/autodiff/1.1.2", 
            "include"
        ],
        language="c++",
        extra_compile_args = ["-O3", "-std=c++17", "-Wall", "-Wextra"], 
    ),
]

setup(
    name="PIMC",
    version="1.2",
    author="Michael Hütter",
    author_email="",
    description="PIMC code",
    ext_modules=ext_modules,
    install_requires=["pybind11"],
)
