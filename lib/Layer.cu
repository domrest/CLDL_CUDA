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



__global__ void gpu_setLearningRate(Neuron** n, double _learningRate){
    //int i = threadIdx.x;
    *n[0]->learningRate = _learningRate;
}

__host__ Layer::Layer(int _nNeurons, int _nInputs){
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron

    neurons = (Neuron**) (malloc(sizeof(Neuron) * nNeurons));
    for (int i=0; i<nNeurons; i++){
       neurons[i] = new Neuron(nInputs);
    }

    cudaMalloc( (void**) &gpu_neurons, sizeof(Neuron)*nNeurons);
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron)*nNeurons, cudaMemcpyHostToDevice);
}

/*__host__ Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    free(neurons);
    cudaFree(gpu_neurons);
}*/


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
    gpu_setLearningRate<<<1,1>>>(gpu_neurons, learningRate);
    //neurons[0]->setLearningRate(0.1);
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

//TODO setInputs
//__host__ void Layer::setInputs(const double *_inputs) {
//    inputs = _inputs;
//    gpu_setInputs<<<1,nNeurons>>>(gpu_neurons, inputs);
//}

//TODO propInputs

//TODO calcOutputs

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

//__host__ void Layer::setForwardError(double _leadForwardError){
//    /*this is only for the first layer*/
//    leadForwardError=_leadForwardError;
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->setForwardError(leadForwardError);
//    }
//}

//TODO setInputs

//__host__ void Layer::propErrorForward(int _index, double _value){
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->propErrorForward(_index, _value);
//    }
//}

//TODO calcForwardError

//TODO getForwardError

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
    return (neurons[_neuronIndex]);
}

//TODO getGlobalError

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
