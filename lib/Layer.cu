#include "cldl/Layer.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>
#include <fstream>

#define MAX_BLOCKSIZE 1024


// GPU FUNCTIONS //

__global__ void gpu_setLearningRate(Neuron* n, double _learningRate) {
    int i = threadIdx.x;
    *n[i].learningRate = _learningRate;
}

__global__ void gpu_setInputs(Neuron* n, const double *list) {
    int i = threadIdx.x;
    int j = blockIdx.x * (threadIdx.x * threadIdx.y);
    printf("%d \n", list[i]);
    n[0].inputs[i] = list[i];
}

__global__ void gpu_setForwardError(Neuron*n, double _leadForwardError) {
    int i = threadIdx.x;
    *n[i].forwardError = _leadForwardError;
}


// HOST FUNCTIONS //

__host__ Layer::Layer(int _nNeurons, int _nInputs){
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron

    neurons = (Neuron*) (malloc(sizeof(Neuron) * nNeurons));
    for (int i=0; i<nNeurons; i++){
        Neuron* j = new Neuron(nInputs);
        neurons[i] = *j;
    }

    cudaMalloc( (void**) &gpu_neurons, sizeof(Neuron)*nNeurons);
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron)*nNeurons, cudaMemcpyHostToDevice);
}

__host__ Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete &neurons[i];
    }
    free(neurons);
    cudaFree(gpu_neurons);
}


//*************************************************************************************
//initialisation:
//*************************************************************************************

//__host__ void Layer::initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
//    myLayerIndex = _layerIndex;
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->initNeuron(i, myLayerIndex, _wim, _bim, _am);
//    }
//}


__host__ void Layer::setlearningRate(double _learningRate){
    learningRate=_learningRate;
    gpu_setLearningRate<<<1,nNeurons>>>(gpu_neurons, learningRate);
    cudaDeviceSynchronize();
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

//TODO setInputs
__host__ void Layer::setInputs(const double *_inputs) {
    inputs = _inputs;
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of a block
    int B = std::ceil(nThreads/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setInputs<<<B,T>>>(gpu_neurons, inputs);
    cudaDeviceSynchronize();
}

//TODO propInputs

//TODO calcOutputs

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

__host__ void Layer::setForwardError(double _leadForwardError){
    /*this is only for the first layer*/
    leadForwardError=_leadForwardError;
    gpu_setForwardError<<<1,nNeurons>>>(gpu_neurons, leadForwardError);
    cudaDeviceSynchronize();
}

//__host__ void Layer::propErrorForward(int _index, double _value){
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->propErrorForward(_index, _value);
//    }
//}

//TODO calcForwardError

__host__ double Layer::getForwardError(int _neuronIndex){
    return (neurons[_neuronIndex].getForwardError());
}

//*************************************************************************************
//back propagation of error:
//*************************************************************************************

//TODO setBackwardError

//TODO propErrorBackward

//TODO getBackwardError

//*************************************************************************************
//mid propagation of error:
//*************************************************************************************

//TODO setMidError

//TODO calcMidError

//TODO getMidError

//TODO propMidErrorForward

//TODO propMidErrorBackward

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

//TODO getGradient

//*************************************************************************************
//learning:
//*************************************************************************************

//TODO setErrorCoeff

//TODO updateWeights

//*************************************************************************************
//global settings
//*************************************************************************************

//TODO setGlobalError

//TODO setEchoError

//TODO getEchoError

//TODO echoErrorBackward

//TODO echoErrorForward

//TODO calcEchoError

//*************************************************************************************
//local backpropagation of error
//*************************************************************************************

//TODO setLocalError

//TODO propGlobalErrorBackwardLocally

//TODO getLocalError

//*************************************************************************************
//getters:
//*************************************************************************************

__host__ Neuron* Layer::getNeuron(int _neuronIndex){
    return (&neurons[_neuronIndex]);
}

/*__host__ double Layer::getGlobalError(int _neuronIndex){
    return (neurons[_neuronIndex].getGlobalError());
}*/

//TODO getSumOutput

//TODO getWeights

//TODO getInitWeight

//TODO getWeightChange

//TODO getWeightDistance

__host__ int Layer::getnNeurons(){
    return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

//TODO saveWeights

//TODO snapWeights

//TODO printLayer
