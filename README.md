# GPU Accelerated Closed-Loop Deep Learning

 This is a GPU-accelerated version of the [Closed-Loop Deep Learning](https://github.com/Sama-Darya/CLDL) library.
 
 Multithreaded processing using a CUDA-enabled GPU allows for much more complex CLDL networks.
 
# Prerequisites

 A CUDA-enabled GPU is required to use this library.
 
 The CUDA developer toolkit is required to compile and run the library.
 
 Install instructions for Windows can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
 
 Install instructions for Linux can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## Building
CLDL_CUDA uses cmake. Just enter the CLDL_CUDA directory from the root:
- ``cd CLDL_CUDA``

and type:
- ``mkdir build && cd build``
- ``cmake ..``
- ``make``

## Test suite:
A gtest test suite is included in the build/gtest directory. The executable Google_Tests_run will be generated automatically when building CLDL_CUDA. Run the tests by doing:
- ``cd build/gtest``
- ``./Google_Tests_run``

## License

GNU GENERAL PUBLIC LICENSE

Version 3, 29 June 2007
