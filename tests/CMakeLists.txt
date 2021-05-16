cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)


project(CLDL LANGUAGES CUDA CXX)

project(tests)

add_executable(tests main.cu)

target_link_libraries(tests PRIVATE CLDL)
