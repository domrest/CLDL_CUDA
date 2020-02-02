#pragma once

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

#include "Layer.h"

class Net {
public:
    // initialisation:
    Net(int _nLayers, int *_nNeurons, int _nInputs);
    ~Net();
    void initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
    void setLearningRate(double _learningRate);

    //forward propagation of inputs:
    void setInputs(const double *_inputs);
    void propInputs();

    //->->forward->-> propagation of error:
    void setErrorAtInput(double _leadForwardError); //only for the first layer
    void propErrorForward();

    //back propagation of error
    void setBackwardError(double _leadError);
    void propErrorBackward();

    //MID propagation of error
    void setMidError(int _layerIndex, double _leadMidError);
    void propMidErrorForward();
    void propMidErrorBackward();

    //learning:
    void setErrorCoeff(double _globalCoeff, double _backwardsCoeff, double _midCoeff, double _forwardCoeff);
    void updateWeights();

    //global settings
    void setGlobalError(double _globalError);

    // getters:
    Layer *getLayer(int _layerIndex);
    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    int getnLayers();
    int getnInputs();
    double getWeightDistance();
    double getLayerWeightDistance(int _layerIndex);
    double getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);
    int getnNeurons();

    //saving and inspecting
    void saveWeights();
    void snapWeights();
    void printNetwork();

private:
    // initialisation:
    int nLayers = 0;
    int nNeurons = 0;
    int nWeights = 0;
    int nInputs = 0;
    int nOutputs = 0;
    double learningRate = 0;
    Layer **layers = 0;

    //forward propagation of inputs:
    const double *inputs = 0;

    //forward propagation of error:
    double leadForwardError = 0;

    //back propagation of error
    double theLeadError = 0;

    //mid propagation of error
    int midLayerIndex = 0;
    double theLeadMidError = 0;

    //learning
    double backwardsCoeff = 0;
    double midCoeff = 0;
    double forwardCoeff = 0;
    double globalCoeff = 0;

    //global settings
    double globalError = 0;
};
