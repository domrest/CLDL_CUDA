#include "cldl/Net.h"
#include "gtest/gtest.h"

using namespace std;

TEST(NetTest, testNetConstructor) {
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

TEST(NetTest, testNetSetLearningRate) {
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

TEST(NetTest, testNetSetInputs) {
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

TEST(NetTest, testNetPropInputs) {
    constexpr int nLayers = 3;
    int nNeurons[nLayers] = {3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 4;
    double inputs[nInputs] = {4,3,2,1};
    double weights[nInputs] = {1,2,3,4};
    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ZEROS, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setInputs(inputs);
    net->setWeights(weights);

    net->propInputs();

    Layer *l;
    l = net->getLayer(0);
    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_EQ(l->getnNeurons(),3);
    ASSERT_EQ(n->getNInputs(), 4);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.04983399731);
    n = l->getNeuron(1);
    ASSERT_EQ(n->getNInputs(), 4);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.04983399731);
    n = l->getNeuron(2);
    ASSERT_EQ(n->getNInputs(), 4);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.04983399731);
    n = l->getNeuron(3);
    ASSERT_EQ(n->getNInputs(), 0);

    l = net->getLayer(1);
    n = l->getNeuron(0);
    ASSERT_EQ(l->getnNeurons(),2);
    ASSERT_EQ(n->getNInputs(), 3);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.07419901436);

    l = net->getLayer(2);
    n = l->getNeuron(0);
    ASSERT_EQ(l->getnNeurons(),1);
    ASSERT_EQ(n->getNInputs(), 2);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.05542061115);
}

TEST(NetTest, testNetSetBackwardError) {
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

TEST(NetTest, testPropErrorBackward) {
    constexpr int nLayers = 4;
    int nNeurons[nLayers] = {4,3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 10;
    double inputs[nInputs] = {1,2,3,4,5,6,7,8,9,10};
    double weights[nInputs] = {1,2,3,4,5,6,7,8,9,10};
    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->setInputs(inputs);
    net->setWeights(weights);

    net->setBackwardError(2.0);
    net->propErrorBackward();

    Layer *l;
    l = net->getLayer(3);   //final layer
    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_EQ(n->getWeight(1), 2);
    ASSERT_EQ(l->leadBackwardError, 2.0);
    ASSERT_EQ(l->getBackwardError(0), 0.5);

    Layer *l2;
    l2 = net->getLayer(2);   //second last layer
    ASSERT_EQ(l2->getBackwardError(0), 0.125);
    ASSERT_EQ(l2->getBackwardError(1), 0.25);

    Layer *l3;
    l3 = net->getLayer(1);  //third last layer/second layer
    ASSERT_EQ(l3->getBackwardError(0), 0.09375);
    ASSERT_EQ(l3->getBackwardError(1), 0.1875);
    ASSERT_EQ(l3->getBackwardError(2), 0.28125);

    Layer *l4;
    l4 = net->getLayer(0);  //first layer
    ASSERT_EQ(l4->getBackwardError(3),0.5625);
}

TEST(NetTest, testNetupdateWeights) {
    constexpr int nLayers = 3;
    int nNeurons[nLayers] = {3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 4;
    double inputs[nInputs] = {4,3,2,1};
    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(0.1);
    net->setErrorCoeff(0,1,0,0,0,0);
    net->setInputs(inputs);
    net->propInputs();
    net->setBackwardError(0.1);
    net->propErrorBackward();
    net->updateWeights();

    Layer *l;
    l = net->getLayer(0);
    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.024979187);

    l = net->getLayer(1);
    n = l->getNeuron(0);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.018725628);

    l = net->getLayer(2);
    n = l->getNeuron(0);
    ASSERT_FLOAT_EQ(n->getOutput(), 0.009361719);
    ASSERT_FLOAT_EQ(n->getBackwardError(), 0.02499123582);
    ASSERT_FLOAT_EQ(n->getWeight(0), 1.0000467977);
}

TEST(NetTest, testNetSetErrorCoeff) {
    constexpr int nLayers = 5;
    int nNeurons[nLayers] = {5, 4, 3, 2, 1};
    int *nNeuronsP = nNeurons;
    constexpr int nInputs = 10;

    Net *net;
    net = new Net(nLayers, nNeuronsP, nInputs);
    net->setErrorCoeff(0,1,0,0,0,0);

    Layer *l;
    l = net->getLayer(4);

    Neuron *n;
    n = l->getNeuron(0);

    ASSERT_EQ(n->getBackwardsCoeff(), 1.0);
}
