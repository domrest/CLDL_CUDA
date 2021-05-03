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
    device_setLearningRate(&n[i], _learningRate);
}

__global__ void gpu_setInputs(Neuron* n, double *list, int nNeurons) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index
    if(j < nNeurons)
        n[j].inputs[i] = list[i];
}

__global__ void gpu_setWeights(Neuron* n, double *list, int nNeurons) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index
    if(j < nNeurons)
        n[j].weights[i] = list[i];
}

__global__ void gpu_calcErrorWeightProductSum(Neuron* n, int nNeurons, int nInputs, double* sumlist) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index

    if(i < nInputs && j < nNeurons)
        n[j].ErrorWeightProducts[i] = n[j].weights[i] * (*n[j].backwardError);
    __syncthreads();

    if (blockIdx.x == 0) {
        double sum = 0.0;
        for (int a = 0; a < nNeurons; a++) {
            sum += n[a].ErrorWeightProducts[i];
        }
        sumlist[i] = sum;
    }
}

/*__global__ void gpu_setForwardError(Neuron*n, double _leadForwardError) {
    int i = threadIdx.x;
    *n[i].forwardError = _leadForwardError;
}*/

__global__ void gpu_setBackwardError(Neuron*n, double _leadBackwardError) {
    int i = threadIdx.x;
    double leadBackwardError = _leadBackwardError;
    device_setBackwardError(leadBackwardError, &n[i]);
}

__global__ void gpu_calcOutputsContinued(Neuron* neurons, int* layerHasReported){
    device_calcOutputCont(&neurons[threadIdx.x], layerHasReported);
}

__global__ void gpu_calcOutputs(Neuron* neurons){
    device_calcOutput(&neurons[blockIdx.x]);
}

__global__ void gpu_propErrorBackwards(Neuron *n, double* _sumList) {
    int i = threadIdx.x;
    double* sumList = _sumList;
    device_propErrorBackward(sumList[i], &n[i]);
}

__global__ void gpu_setErrorCoeff(Neuron *n, double _globalCoeff, double _backwardsCoeff,
                                  double _midCoeff, double _forwardCoeff,
                                 double _localCoeff, double _echoCoeff) {
    int i = threadIdx.x;
    *n[i].backwardsCoeff = _backwardsCoeff;
    *n[i].midCoeff = _midCoeff;
    *n[i].forwardCoeff =_forwardCoeff;
    *n[i].globalCoeff = _globalCoeff;
    *n[i].localCoeff = _localCoeff;
    *n[i].echoCoeff= _echoCoeff;
}

__global__ void gpu_updateWeights(Neuron *n, int nNeurons){
    int i = threadIdx.x;    //Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y;  //Neuron index
    //double force = 1;
    if (j<nNeurons) {
        n[j].weights[i] += (*n[j].learningRate) * n[j].inputs[i] * (*n[j].backwardError); // * force;
    }
}

__global__ void gpu_getOutputs(Neuron* n, double* _outputs){
    int x = threadIdx.x;
    _outputs[x] = *n[x].output;
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

    cudaMalloc((void**) &gpu_sumlist, sizeof(double)*_nInputs);
    cudaMalloc((void**) &gpu_weights, sizeof(double)*nInputs);
    cudaMalloc( (void**) &gpu_inputs, sizeof(double)*nInputs);
    cudaMalloc( (void**) &gpu_neurons, sizeof(Neuron)*nNeurons);
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron)*nNeurons, cudaMemcpyHostToDevice);
}

__host__ Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        neurons[i].~Neuron();
    }
    free(neurons);
    cudaFree(gpu_inputs);
    cudaFree(gpu_neurons);
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

__host__ void Layer::initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    myLayerIndex = _layerIndex;
    for (int i=0; i<nNeurons; i++){
        neurons[i].initNeuron(i, myLayerIndex, _wim, _bim, _am);
    }
}

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

__host__ void Layer::propInputs(double* _gpu_InputOutputs) {
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setInputs<<<B,T>>>(gpu_neurons, _gpu_InputOutputs, nNeurons);
    cudaDeviceSynchronize();
}

__host__ void Layer::calcOutputs(){
    // block id gets neuron
    int* _layerHasReported;
    gpu_allocateInt(&_layerHasReported, 0);
    cudaMemcpy(_layerHasReported, &layerHasReported, sizeof(int), cudaMemcpyHostToDevice);

    gpu_calcOutputs<<<nNeurons, 1>>>(gpu_neurons);
    cudaDeviceSynchronize();
    gpu_calcOutputsContinued<<<1,nNeurons>>>(gpu_neurons, _layerHasReported);
    cudaDeviceSynchronize();

    cudaMemcpy(&layerHasReported, _layerHasReported, sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ double* Layer::getOutputs(){
    double* _outputs;
    cudaMalloc(&_outputs, sizeof(double)*getnNeurons());
    gpu_getOutputs<<<1, getnNeurons()>>>(gpu_neurons, _outputs);
    return _outputs;
//    return (neurons[_neuronIndex]->getOutput());
}

__host__ double Layer::getOutput(int _neuronIndex) {
    return (neurons[_neuronIndex].getOutput());
}

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

/*__host__ void Layer::setForwardError(double _leadForwardError){
    this is only for the first layer
    leadForwardError=_leadForwardError;
    gpu_setForwardError<<<1,nNeurons>>>(gpu_neurons, leadForwardError);
    cudaDeviceSynchronize();
}*/

//__host__ void Layer::propErrorForward(int _index, double _value){
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->propErrorForward(_index, _value);
//    }
//}

/*__host__ double Layer::getForwardError(int _neuronIndex){
    return (neurons[_neuronIndex].getForwardError());
}*/

//*************************************************************************************
//back propagation of error:
//*************************************************************************************

__host__ void Layer::setBackwardError(double _leadBackwardError) {
    leadBackwardError = _leadBackwardError;
    gpu_setBackwardError<<<1,nNeurons>>>(gpu_neurons, leadBackwardError);
    cudaDeviceSynchronize();
}

__host__ double* Layer::calcErrorWeightProductSum() {
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_calcErrorWeightProductSum<<<B,T>>>(gpu_neurons, nNeurons, nInputs, gpu_sumlist);
    cudaDeviceSynchronize();
    return gpu_sumlist;
}

__host__ double Layer::getErrorWeightProductSum(int index) {
    double _sum = 0.0;
    double* sum = gpu_sumlist + index;
    cudaMemcpy(&_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);
    return _sum;
}

__host__ void Layer::propErrorBackward(double* _sumList) {
    gpu_propErrorBackwards<<<1,nNeurons>>>(gpu_neurons, _sumList);
    cudaDeviceSynchronize();
}

__host__ double Layer::getBackwardError(int _neuronIndex){
    return (neurons[_neuronIndex].getBackwardError());
}

//*************************************************************************************
//learning:
//*************************************************************************************

__host__ void Layer::setErrorCoeff(double _globalCoeff, double _backwardsCoeff,
                            double _midCoeff, double _forwardCoeff,
                            double _localCoeff, double  _echoCoeff) {
    gpu_setErrorCoeff<<<1,nNeurons>>>(gpu_neurons, _globalCoeff, _backwardsCoeff,
                                      _midCoeff, _forwardCoeff, _localCoeff, _echoCoeff);
    cudaDeviceSynchronize();
}

__host__ void Layer::updateWeights() {
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions

    gpu_updateWeights<<<B,T>>>(gpu_neurons, nNeurons);
    cudaDeviceSynchronize();
}

//this method is for testing only
__host__ void Layer::setWeights(double* _weightsList) {
    cudaMemcpy(gpu_weights, _weightsList, sizeof(double)*nInputs,cudaMemcpyHostToDevice);
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setWeights<<<B,T>>>(gpu_neurons, gpu_weights, nNeurons);
}

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

__host__ void Layer::printWeights(FILE* weights) {
    for (int i=0;i<nNeurons;i++) {
        for (int j=0;j<nInputs;j++) {
            fprintf(weights,"%f, ", neurons[i].getWeight(j));
        }
        fprintf(weights,"\n");
    }
    fprintf(weights,"\n");
}
