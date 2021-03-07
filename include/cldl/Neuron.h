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

    /**
        * Options for method of initialising biases
        * 0 for initialising all weights to zero
        * 1 for initialising all weights to one
        * 2 for initialising all weights to a random value between 0 and 1
        */
    enum biasInitMethod { B_NONE = 0, B_RANDOM = 1 };

    /**
     * Options for method of initialising weights
     * 0 for initialising all weights to zero
     * 1 for initialising all weights to one
     * 2 for initialising all weights to a random value between 0 and 1
     */
    enum weightInitMethod { W_ZEROS = 0, W_ONES = 1, W_RANDOM = 2 };

    /**
     * Options for activation functions of the neuron
     * 0 for using the logistic function
     * 1 for using the hyperbolic tan function
     * 2 for unity function (no activation)
     */
    enum actMethod {Act_Sigmoid = 0, Act_Tanh = 1, Act_NONE = 2};

    /**
     * Options for choosing an error to monitor the gradient of
     * 0 for monitoring the error that propagates backward
     * 1 for monitoring the error that propagates from the middle and bilaterally
     * 2 for monitoring the error that propagates forward
     */
    enum whichError {onBackwardError = 0, onMidError = 1, onForwardError = 2};

    /**
     * Initialises the neuron with the given methods for weight/bias initialisation and for activation function.
     * It also specifies the index of the neuron and the index of the layer that contains this neuron.
     * @param _neuronIndex The index of this neuron
     * @param _layerIndex The index of the layer that contains this neuron
     * @param _wim The method of initialising the weights, refer to weightInitMethod for more information
     * @param _bim The method of initialising the biases, refer to biasInitMethod for more information
     * @param _am The function used for activation of neurons, refer to actMethod for more information
     */
    __host__ void initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am);

    //Forward Propagation of inputs:
    __host__ void setInput(int _index,  double _value);
    __host__ void propInputs(int _index,  double _value);
    __host__ double getInput(int index);

    __host__ void setLearningRate(double _learningRate);
    __host__ double getLearningRate();
    __host__ int getNInputs();

    //Forward Propagation of errors:
    __host__ void setForwardError(double _value);
    __host__ double getInputError(int index);
    __host__ void propErrorForward(int _index, double _value);


private:
    // initialisation:
    int *nInputs;
    int *myLayerIndex;
    int *myNeuronIndex;
    double *initialWeights;
    double *learningRate;

    int *iHaveReported;

    //forward propagation of inputs:
    double *inputs;
    double *bias;
    double *sum;
    double *output;

    //forward propagation of error:
    double *inputErrors;
    double *forwardError;

    //back propagation of error
    double *backwardError;

    //mid propagation of error
    double *inputMidErrors;
    double *midError;

    //learning:
    double *backwardsCoeff;
    double *midCoeff;
    double *forwardCoeff;
    double *globalCoeff;
    double *weights;
    double *weightSum;
    double *maxWeight;
    double *minWeight;
    double *weightChange;
    double *weightsDifference;
    int *actMet;

    //global setting
    double *globalError;
    double *localError;
    double *echoCoeff;
    double *localCoeff;

    double *overallError;
    double *echoError;
    double *echoErrors;


};
__global__ void gpu_setValuesInArray(double _value, double* list);
__global__ void gpu_setValueInArray(double _value, int index, double* list);
__global__ void gpu_getSumAndMaxMin(double* sum, double* max_list, double* list_min, double* list, int length);

__host__ void gpu_allocateInt(int** pointer, int value);
__global__ void gpu_setInt(int* pointer, int value);

__host__ void gpu_allocateDouble(double** pointer, double value);
__global__ void gpu_setDouble(double* pointer, double value);
__global__ void gpu_dotProduct(double* list1, double* list2, double* _value, double* _target, int arrayLength);