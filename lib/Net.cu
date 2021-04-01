#include "cldl/Net.h"

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

using namespace std;

__host__ Net::Net(int _nLayers, int* _nNeurons, int _nInputs) {
    nLayers = _nLayers; //no. of layers including inputs and outputs layers
    layers = new Layer*[nLayers];
    int* nNeuronsp = _nNeurons; //no. of neurons in each layer (note this is an array)
    nInputs=_nInputs;   // the no. of inputs to the network (i.e. the first layer)

    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int numNeurons= *nNeuronsp; //no. neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new Layer(numNeurons, nInput);
        nNeurons += numNeurons;
        nWeights += (numNeurons * nInput);
        nInput=numNeurons;
        /*no. inputs to the next layer is equal to the number of neurons
         * in the current layer. */
        nNeuronsp++; //point to the no. of neurons in the next layer
    }
    nOutputs=layers[nLayers-1]->getnNeurons();
    errorGradient= new double[nLayers];
}

//TODO Destructor
/*__host__ Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete &layers[i];
    }
    delete[] layers;
    delete[] errorGradient;
}*/

//TODO initNetwork
/*__host__ void Net::initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nLayers; i++){
        layers[i]->initLayer(i, _wim, _bim, _am);
    }
}*/

__host__ void Net::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nLayers; i++){
        layers[i]->setlearningRate(learningRate);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

__host__ void Net::setInputs(double* _inputs){
    inputs=_inputs;
    layers[0]->setInputs(inputs); //sets the inputs to the first layer only
}

__host__ void Net::propInputs() {
    for (int i=0;i<nLayers-1; i++) {
        layers[i]->calcOutputs();
        double* layerOutputs = layers[i]->getOutput();
        layers[i+1]->propInputs(layerOutputs);
    }
    layers[nLayers-1]->calcOutputs();
}

//*************************************************************************************
//back propagation of error
//*************************************************************************************

__host__ void Net::setBackwardError(double _leadError){
    /* this is only for the final layer */
    theLeadError = _leadError;
    //cout<< "lead Error: " << theLeadError << endl;
    layers[nLayers-1]->setBackwardError(theLeadError);
    /* if the leadError was diff. for each output neuron
     * then it would be implemented in a for-loop */
}

__host__ void Net::propErrorBackward() {
    for (int i = nLayers - 1; i > 0; i--) {
        //double sum = layers[i]->getSum();
        //layers[i-1]->propErrorBackward(sum);
    }
}

//*************************************************************************************
//learning:
//*************************************************************************************

/*void Net::setErrorCoeff(double _globalCoeff, double _backwardsCoeff, double _midCoeff, double _forwardCoeff, double _localCoeff, double  _echoCoeff){
    for (int i=0; i<nLayers; i++){
        layers[i]->setErrorCoeff(_backwardsCoeff);
    }
}*/

void Net::updateWeights(){
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->updateWeights();
    }
}

//*************************************************************************************
// getters:
//*************************************************************************************

//TODO getters

__host__ int Net::getnLayers(){
    return (nLayers);
}

__host__ int Net::getnNeurons(){
    return (nNeurons);
}

__host__ int Net::getnInputs(){
    return (nInputs);
}

__host__ int Net::getnOutputs(){
    return (nOutputs);
}

Layer* Net::getLayer(int _layerIndex){
    assert(_layerIndex<nLayers);
    return (layers[_layerIndex]);
}
