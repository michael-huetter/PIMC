from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'PIMC',
        ['binding.cpp', 'mcmc.cpp', 'Beads.cpp', 'Energy.cpp', 'Potential.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3',  
        ],
        language='c++',
        extra_compile_args = ['-O3', '-std=c++17', '-Wall', '-Wextra'], 
    ),
]

setup(
    name='PIMC',
    version='1.2',
    author='Michael Hütter',
    author_email='',
    description='PIMC code',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
)
