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

__global__ void gpu_setInputs(Neuron* n, double *list, int nNeurons) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index
    if(j < nNeurons)
        n[j].inputs[i] = list[i];
}

__global__ void gpu_setForwardError(Neuron*n, double _leadForwardError) {
    int i = threadIdx.x;
    *n[i].forwardError = _leadForwardError;
}

__global__ void gpu_setBackwardError(Neuron*n, double _leadBackwardError) {
    int i = threadIdx.x;
    *n[i].backwardError = _leadBackwardError;
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

    cudaMalloc( (void**) &gpu_inputs, sizeof(double)*nInputs);
    cudaMalloc( (void**) &gpu_neurons, sizeof(Neuron)*nNeurons);
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron)*nNeurons, cudaMemcpyHostToDevice);
}

__host__ Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete &neurons[i];
    }
    free(neurons);
    cudaFree(gpu_inputs);
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

__host__ void Layer::setInputs(double *_inputs) {
    inputs = _inputs;
    cudaMemcpy(gpu_inputs, inputs, sizeof(double)*nInputs,cudaMemcpyHostToDevice);

    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setInputs<<<B,T>>>(gpu_neurons, gpu_inputs, nNeurons);

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

__host__ void Layer::setBackwardError(double _leadBackwardError) {
    leadBackwardError = _leadBackwardError;
    gpu_setBackwardError<<<1,nNeurons>>>(gpu_neurons, leadBackwardError);
    cudaDeviceSynchronize();
}

//TODO propErrorBackward

__host__ double Layer::getBackwardError(int _neuronIndex){
    return (neurons[_neuronIndex].getBackwardError());
}

//*************************************************************************************
//learning:
//*************************************************************************************

//TODO setErrorCoeff

//TODO updateWeights

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
