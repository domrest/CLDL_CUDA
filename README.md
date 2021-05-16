# GPU Accelerated Closed-Loop Deep Learning

 This is a GPU-accelerated version of the Closed-Loop Deep Learning library.
 
 Multithreaded processing using a CUDA-enabled GPU allows for much more complex CLDL nets.
 
# Prerequisites

 A CUDA-enabled GPU is required to use this library.
 
 The CUDA developer toolkit is required to compile and run the library.
 
 Install instructions for Windows can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
 
 Install instructions for Linux can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

## Building
CLDL_CUDA uses cmake. Just enter the CLDL_CUDA directory from the root:
- ``cd CLDL_CUDA``

and type:
- ``mkdir build && cd build``
- ``cmake ..``
- ``make``

## Test suite:
A gtest test suite is included in the tests directory. The executable tests will be generated automatically when building CLDL. Run the test by doing:
- ``cd tests``
- ``./tests``

## License

GNU GENERAL PUBLIC LICENSE

Version 3, 29 June 2007
