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

#install(TARGETS CLDL
#LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cldl_cuda)

find_package(Threads)

target_link_libraries(CLDL ${CMAKE_THREAD_LIBS_INIT})
