cmake_minimum_required(VERSION 3.14)
project(CLDL LANGUAGES CUDA CXX)

add_subdirectory(tests)
add_subdirectory(gtest)

# packages
find_package(CUDA QUIET REQUIRED)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -rdc=true;)

cuda_add_library(CLDL SHARED
        lib/Neuron.cu
        lib/Layer.cu
        lib/Net.cu)

set_target_properties(
		CLDL
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

install(TARGETS CLDL
LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cldl_cuda)

target_include_directories(CLDL
        PUBLIC
        include/)


cuda_add_library(CLDL_STATIC STATIC
        lib/Neuron.cu
        lib/Layer.cu
        lib/Net.cu)

set_target_properties(
		CLDL_STATIC
		PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
		POSITION_INDEPENDENT_CODE ON)

target_include_directories(CLDL_STATIC
		PUBLIC
		include/)

find_package(Threads)

target_link_libraries(CLDL_STATIC ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(CLDL_STATIC PRIVATE CLDL)

set_target_properties(CLDL_STATIC PROPERTIES
  PUBLIC_HEADER "${PUBLICHEADERS}"
  PRIVATE_HEADER "${PRIVATEHEADERS}")

install(TARGETS CLDL_STATIC
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cldl_cuda)
