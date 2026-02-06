cmake_minimum_required(VERSION 3.18)
project(cufractal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Исходники
add_library(cufractal MODULE
    src/fractal_cpu.cpp
    src/fractal_cuda.cu
    src/bindings.cpp
)
