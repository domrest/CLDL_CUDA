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

    cudaMalloc((void**)&nInputs, sizeof(int));
    cudaMemcpy(nInputs, &_nInputs, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&learningRate, sizeof(double));
    cudaMemcpyFromSymbol(learningRate, 0.0, sizeof(double));

    cudaMalloc((void**)&weights, sizeof(double)*_nInputs);
    cudaMalloc((void**)&initialWeights, sizeof(double)*_nInputs);
    cudaMalloc((void**)&inputs, sizeof(double)*_nInputs);
    cudaMalloc((void**)&inputErrors, sizeof(double)*_nInputs);
    cudaMalloc((void**)&inputMidErrors, sizeof(double)*_nInputs);
    cudaMalloc((void**)&echoErrors, sizeof(double)*_nInputs);

    //cout << "neuron" << endl;

}

__host__ Neuron::~Neuron(){
    cudaFree(nInputs);
    cudaFree(learningRate);

    cudaFree(weights);
    cudaFree(initialWeights);
    cudaFree(inputs);
    cudaFree(inputErrors);
    cudaFree(inputMidErrors);
    cudaFree(echoErrors);
}


//*************************************************************************************
//initialisation:
//*************************************************************************************

//TODO initNeuron

__host__ void Neuron::setLearningRate(double _learningRate){
    cudaMemcpy(learningRate, &_learningRate, sizeof(double), cudaMemcpyHostToDevice);
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
//global kernel:
//*************************************************************************************

__global__ static void gpu_setValuesInArray(double _value, double* list){
    list[threadIdx.x] = _value;
}

__global__ static void gpu_setValueInArray(double _value, int index, double* list){
    list[index] = _value;
}
