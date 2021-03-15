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

// TODO private variables


__host__ Layer::Layer(int _nNeurons, int _nInputs){
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new Neuron*[nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new Neuron(nInputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs */
}

__host__ Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}


//*************************************************************************************
//initialisation:
//*************************************************************************************

__host__ void Layer::initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    myLayerIndex = _layerIndex;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initNeuron(i, myLayerIndex, _wim, _bim, _am);
    }
}




__host__ void Layer::setlearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(learningRate);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

//TODO setInputs

//TODO propInputs

//TODO calcOutputs

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

//__host__ void Layer::setForwardError(double _leadForwardError) {
//    double** pointerList = new double*[nNeurons];
//    int* nInputList = new int[nNeurons];
//
//    for (int i=0; i<nNeurons; i++){
//        pointerList[i] = neurons[i]->getInputErrorPointer();
//        nInputList[i] = neurons[1]->getNInputs();
//    }
//
//}

//__global__ void gpu_setForwardError(double _leadForwardError, double** pointerList, int* nInputList) {
//
//}
//
//__host__ void Layer::setForwardError(double _leadForwardError){
//    /*this is only for the first layer*/
//    leadForwardError=_leadForwardError;
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->setForwardError(leadForwardError);
//    }
//}

//TODO setInputs

__host__ void Layer::propErrorForward(int _index, double _value){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->propErrorForward(_index, _value);
    }
}

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
    assert(_neuronIndex < nNeurons);
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
