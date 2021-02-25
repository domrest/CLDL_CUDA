cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
SET(CUDA_SEPARABLE_COMPILATION ON)

project(CLDL LANGUAGES CUDA CXX)

add_library(CLDL
#        lib/Layer.cpp
#        lib/Net.cpp
        lib/Neuron.cu)

target_include_directories(CLDL PUBLIC include)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")



add_subdirectory(tests)
add_subdirectory(gtest)