#include "cldl/Net.h"
#include "gtest/gtest.h"

using namespace std;

TEST(CUDATest, testNetConstructor) {
    constexpr int nLayers = 5;
    int nNeurons[nLayers] = {5,4,3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 10;

    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);

    ASSERT_EQ(net->getnInputs(), 10);
    ASSERT_EQ(net->getnLayers(), 5);
    ASSERT_EQ(net->getnNeurons(), 15);
    ASSERT_EQ(net->getnOutputs(), 1);

    Layer *l;
    l = net->getLayer(0);
    ASSERT_EQ(l->getnNeurons(), 5);

    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_EQ(n->getNInputs(), 10);

    l = net->getLayer(1);
    ASSERT_EQ(l->getnNeurons(), 4);

    n = l->getNeuron(3);
    ASSERT_EQ(n->getNInputs(), 5);
}

//TODO testDestructor

//TODO testInitNetwork

TEST(CUDATest, testNetSetLearningRate) {
    constexpr int nLayers = 5;
    int nNeurons[nLayers] = {5,4,3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 10;

    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->setLearningRate(0.1);

    Layer *l;
    l = net->getLayer(2);

    Neuron *n;
    n = l->getNeuron(2);
    ASSERT_EQ(n->getLearningRate(), 0.1);
}

TEST(CUDATest, testNetSetInputs) {
    constexpr int nLayers = 5;
    int nNeurons[nLayers] = {5,4,3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 10;
    double inputs[nInputs] = {1,2,3,4,5,6,7,8,9,10};
    double* inputsp = inputs;

    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->setInputs(inputsp);

    Layer *l;
    l = net->getLayer(0);

    Neuron *n;
    n = l->getNeuron(2);
    ASSERT_EQ(n->getInput(5), 6);
}

//TODO testNetPropInputs

TEST(CUDATest, testNetSetBackwardError) {
    constexpr int nLayers = 5;
    int nNeurons[nLayers] = {5, 4, 3, 2, 1};
    int *nNeuronsP = nNeurons;
    constexpr int nInputs = 10;

    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->setBackwardError(0.1);

    Layer *l;
    l = net->getLayer(4);

    Neuron *n;
    n = l->getNeuron(0);

    ASSERT_EQ(l->leadBackwardError, 0.1);
    ASSERT_EQ(n->getBackwardError(), 0.025);
}

//TODO testNetPropErrorBackward

//TODO testNetUpdateWeights

//TODO testNetSetErrorCoeff
