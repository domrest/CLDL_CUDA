cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

project(CLDL LANGUAGES CUDA CXX)

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)

# 'Google_test' is the subproject name
project(gtest)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} ../include)


# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Google_Tests_run NeuronTest.cu)
target_link_libraries(Google_Tests_run PRIVATE CLDL gtest gtest_main)

