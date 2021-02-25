#include "cldl/Neuron.h"

#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <numeric>
#include <vector>

using namespace std;

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

__host__ Neuron::Neuron(int _nInputs)
{
    // initialisation
    gpu_allocateInt(&nInputs, _nInputs);
    gpu_allocateInt(&myLayerIndex, 0);
    gpu_allocateInt(&myNeuronIndex, 0);
    cudaMalloc((void**)&initialWeights, sizeof(double)*_nInputs);
    gpu_allocateDouble(&learningRate, 0.0);

    gpu_allocateInt(&iHaveReported, 0);

    // forward propagation of inputs
    cudaMalloc((void**)&inputs, sizeof(double)*_nInputs);
    gpu_allocateDouble(&bias, 0.0);
    gpu_allocateDouble(&sum, 0.0);
    gpu_allocateDouble(&output, 0.0);

    // forward propagation of error
    cudaMalloc((void**)&inputErrors, sizeof(double)*_nInputs);
    gpu_allocateDouble(&forwardError, 0.0);


    // back propagation of error
    gpu_allocateDouble(&backwardError, 0.0);

    // mid propagation of error
    cudaMalloc((void**)&inputMidErrors, sizeof(double)*_nInputs);
    gpu_allocateDouble(&midError, 0.0);


    //
    // learning variables
    //
    gpu_allocateDouble(&backwardsCoeff, 0.0);
    gpu_allocateDouble(&midCoeff, 0.0);
    gpu_allocateDouble(&forwardCoeff, 0.0);
    gpu_allocateDouble(&globalCoeff, 0.0);

    cudaMalloc((void**)&weights, sizeof(double)*_nInputs);

    gpu_allocateDouble(&weightSum, 0.0);
    gpu_allocateDouble(&maxWeight, 1.0);
    gpu_allocateDouble(&minWeight, 1.0);
    gpu_allocateDouble(&weightChange, 0.0);
    gpu_allocateDouble(&weightsDifference, 0.0);
    gpu_allocateInt(&actMet, 0);

    //
    // learning variables end
    //

    cudaMalloc((void**)&echoErrors, sizeof(double)*_nInputs);

    //cout << "neuron" << endl;

}

__host__ Neuron::~Neuron(){
    //initialisation
    cudaFree(nInputs);
    cudaFree(learningRate);
    cudaFree(myLayerIndex);
    cudaFree(initialWeights);
    cudaFree(myNeuronIndex);

    cudaFree(iHaveReported);

    // forward propagation of inputs
    cudaFree(inputs);
    cudaFree(bias);
    cudaFree(sum);
    cudaFree(output);

    // forward propagation of error
    cudaFree(inputErrors);
    cudaFree(forwardError);

    // back propagation of error
    cudaFree(backwardError);

    // mid propagation of error
    cudaFree(inputMidErrors);
    cudaFree(midError);


    //learning
    cudaFree(backwardsCoeff);
    cudaFree(midCoeff);
    cudaFree(forwardCoeff);
    cudaFree(globalCoeff);
    cudaFree(weights);
    cudaFree(weightSum);
    cudaFree(maxWeight);
    cudaFree(minWeight);
    cudaFree(weightChange);
    cudaFree(weightsDifference);
    cudaFree(actMet);



    cudaFree(echoErrors);
}


//*************************************************************************************
//initialisation:
//*************************************************************************************

//TODO test init neuron
__host__ void Neuron::initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am){
    cudaMemcpy(myLayerIndex, &_layerIndex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(myNeuronIndex, &_neuronIndex, sizeof(int), cudaMemcpyHostToDevice);
    switch(_wim) {
        case W_ZEROS:
            gpu_setValuesInArray<<<1,getNInputs()>>>(0, weights);
            break;
        case W_ONES:
            gpu_setValuesInArray<<<1,getNInputs()>>>(1, weights);
            break;
        case W_RANDOM:
            //TODO set the random
//            weights[i] = (((double) rand() / (RAND_MAX))); //* 2) -1;
            break;
            //cout << " Neuron: weight is: " << weights[i] << endl;
            /* rand function generates a random function between
             * 0 and RAND_MAX, after the devision the weights are
             * set to a value between 0 and 1 */
    }
    cudaMemcpy(initialWeights, weights, sizeof(double)*getNInputs(), cudaMemcpyDeviceToDevice);

    gpu_setDouble<<<1,1>>>(weightSum, 0);
    gpu_getSumAndMaxMin<<<1,1>>>(weightSum, maxWeight, minWeight, weights, getNInputs());

    switch (_bim){
        case B_NONE:
            gpu_setDouble<<<1,1>>>(bias, 0.0);
            break;
        case B_RANDOM:
            gpu_setDouble<<<1,1>>>(bias, ((double)rand()/RAND_MAX));
            break;
    }
    switch(_am){
        case Act_Sigmoid:
            gpu_setInt<<<1,1>>>(actMet, 0);
            break;
        case Act_Tanh:
            gpu_setInt<<<1,1>>>(actMet, 1);
            break;
        case Act_NONE:
            gpu_setInt<<<1,1>>>(actMet, 2);
            break;
    }
}

__host__ void Neuron::setLearningRate(double _learningRate){
    gpu_setDouble<<<1,1>>>(learningRate, _learningRate);
}

__host__ double Neuron::getLearningRate() {
    double _learningRate;
    cudaMemcpy(&_learningRate, learningRate, sizeof(double), cudaMemcpyDeviceToHost);
    return _learningRate;
}


//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

//TODO setInput

//TODO propInputs

//TODO calcOutput

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

__host__ void Neuron::setForwardError(double _value) {
    gpu_setValuesInArray<<<1, getNInputs()>>>(_value, inputErrors);
}

__host__ double Neuron::getInputError(int index) {
    double _inputError = 0.0;
    assert(index < getNInputs());

    double* inputError = inputErrors + index;
    cudaMemcpy(&_inputError, inputError, sizeof(double), cudaMemcpyDeviceToHost);
    return _inputError;
}

__host__ void Neuron::propErrorForward(int _index, double _value){
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputErrors);
}


//TODO calcForwardError

//TODO getForwardError

//*************************************************************************************
//back propagation of error
//*************************************************************************************

//TODO setBackwardError

//TODO propErrorBackward

//TODO getBackwardError

//TODO getEchoError

//TODO echoErrorBackward

//*************************************************************************************
//MID propagation of error
//*************************************************************************************

//TODO setMidError

//TODO calcMidError

//TODO getMidError

//TODO propMidErrorForward

//TODO propMidErrorBackward

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

//TODO getError

//*************************************************************************************
//learning
//*************************************************************************************

//TODO setErrorCoeff

//TODO updateWeights

//TODO doActivation

//TODO doActivationPrime

//*************************************************************************************
//global settings
//*************************************************************************************

//TODO setGlobalError

//TODO getGlobalError

//TODO setEchoError

//TODO echoErrorForward

//TODO calcEchoError

//*************************************************************************************
//local backpropagation of error
//*************************************************************************************

//TODO setLocalError

//TODO propGlobalErrorBackwardLocally

//TODO getLocalError

//*************************************************************************************
// getters:
//*************************************************************************************

//TODO getOutput

//TODO getSumOutput

//TODO getMaxWeight

//TODO getMinWeight

//TODO getSumWeight

//TODO getWeightChange

//TODO getWeightDistance

__host__ int Neuron::getNInputs(){
    int _nInputs=0;
    cudaMemcpy(&_nInputs, nInputs, sizeof(int), cudaMemcpyDeviceToHost);
    return _nInputs;
}

//TODO getWeights

//TODO getInitWeights

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

//TODO saveWeights

//TODO printNeuron


//*************************************************************************************
//global CUDA kernels:
//*************************************************************************************

__global__ void gpu_setValuesInArray(double _value, double* list){
    list[threadIdx.x] = _value;
}

__global__ void gpu_setValueInArray(double _value, int index, double* list){
    list[index] = _value;
}

__global__ void gpu_getSumAndMaxMin(double* sum, double* max_list, double* list_min, double* list, int length){
    for (int i=0; i<length; i++){
        *sum = *sum + fabs(list[i]);
        *max_list = max(*max_list, list[i]);
        *list_min = min(*list_min, list[i]);
    }
}

__host__ void gpu_allocateInt(int** pointer, int value){
    cudaMalloc((void**)pointer, sizeof(int));
    gpu_setInt<<<1,1>>>(*pointer, value);
}
__global__ void gpu_setInt(int* pointer, int value) {
    *pointer = value;
}
__host__ void gpu_allocateDouble(double** pointer, double value){
    cudaMalloc((void**)pointer, sizeof(double));
    gpu_setDouble<<<1,1>>>(*pointer, value);
}
__global__ void gpu_setDouble(double* pointer, double value){
    *pointer = value;
}