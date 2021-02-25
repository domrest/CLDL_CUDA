#pragma once

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <math.h>
#include <fstream>
#include <string>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#define CUDA_HOSTDEV __host__ __device__

using namespace std;

/**
 * This is the class for creating neurons inside the Layer class.
 * This is the building block class of the network.
 */

class Neuron {
public:

    /**
     * Constructor for the Neuron class: it initialises a neuron with specific number fo inputs to that neuron
     * @param _nInputs
     */
    Neuron(int _nInputs);
    /**
     * Destructor
     * De-allocated any memory
     */
    ~Neuron();

    __host__ int getNInputs();

private:
    // initialisation:
    int *nInputs;
    double *initialWeights;

    //forward propagation of inputs:
    double *inputs;

    //forward propagation of error:
    double *inputErrors;

    //back propagation of error

    //mid propagation of error
    double *inputMidErrors;

    //learning:
    double *weights;

    //global setting
    double *echoErrors;

};
