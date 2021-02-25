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


    __host__ void setLearningRate(double _learningRate);
    __host__ double getLearningRate();
    __host__ int getNInputs();

    __host__ void setForwardError(double _value);
    __host__ double getInputError(int index);
    __host__ void propErrorForward(int _index, double _value);




private:
    // initialisation:
    int *nInputs;
    double *initialWeights;
    double *learningRate;

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
__global__ static void gpu_setValuesInArray(double _value, double* inputErrors);
__global__ static void gpu_setValueInArray(double _value, int index, double* inputErrors);

