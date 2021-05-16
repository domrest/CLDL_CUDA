cmake_minimum_required(VERSION 3.14)
project(CLDL LANGUAGES CUDA CXX)

add_subdirectory(gtest)

# packages
find_package(CUDA QUIET REQUIRED)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -rdc=true;)

cuda_add_library(CLDL STATIC
        lib/Neuron.cu
        lib/Layer.cu
        lib/Net.cu)

set_target_properties(
		CLDL
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

target_include_directories(CLDL
        PUBLIC
        include/)

find_package(Threads)

target_link_libraries(CLDL curand ${CMAKE_THREAD_LIBS_INIT})

