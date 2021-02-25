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
#define CUDA_HOSTDEV __host__ __device__

using namespace std;

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

__host__ Neuron::Neuron(int _nInputs)
{

    cudaMalloc((void**)&nInputs, sizeof(int));
    cudaMemcpy(nInputs, &_nInputs, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&weights, sizeof(double)*_nInputs);
    cudaMalloc((void**)&initialWeights, sizeof(double)*_nInputs);
    cudaMalloc((void**)&inputs, sizeof(double)*_nInputs);
    cudaMalloc((void**)&inputErrors, sizeof(double)*_nInputs);
    cudaMalloc((void**)&inputMidErrors, sizeof(double)*_nInputs);
    cudaMalloc((void**)&echoErrors, sizeof(double)*_nInputs);

    //cout << "neuron" << endl;

}

__host__ Neuron::~Neuron(){
    cudaFree(weights);
    cudaFree(initialWeights);
    cudaFree(inputs);
    cudaFree(inputErrors);
    cudaFree(inputMidErrors);
    cudaFree(echoErrors);
}

__host__ int Neuron::getNInputs(){
    int _nInputs=0;
    cudaMemcpy(&_nInputs, nInputs, sizeof(int), cudaMemcpyDeviceToHost);
    return _nInputs;
}
