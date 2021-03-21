#include "cldl/Layer.h"
#include "gtest/gtest.h"

using namespace std;

TEST(LayerTest, testLayerConstructor){
    Layer *l;
    l = new Layer(10, 10);
    //Check there are 10 neurons in the layer
    ASSERT_EQ(l->getnNeurons(), 10);

    //Check that neurons have 10 inputs
    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_EQ(n->getNInputs(), 10);
}

//TODO testLayerDestructor

TEST(LayerTest, testLayerSetLearningRate) {
    Layer *l;
    l = new Layer(10, 10);
    l->setlearningRate(0.1);

    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_EQ(n->getLearningRate(), 0.1);
}




TEST(LayerTest, testLayerSetInputs) {
    Layer *l;
    l = new Layer(10, 10);

    double in[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    l->setInputs(in);
    ASSERT_EQ(l->inputs[3], 4.0);

    Neuron *n;
    n = l->getNeuron(3);
    ASSERT_EQ(n->getNInputs(), 10);
    ASSERT_EQ(n->getInput(5), 6.0);
    ASSERT_EQ(n->getInput(2), 3.0);

    Neuron *n2;
    n2 = l->getNeuron(7);
    ASSERT_EQ(n2->getNInputs(), 10);
    ASSERT_EQ(n2->getInput(5), 6.0);
    ASSERT_EQ(n2->getInput(2), 3.0);
}

TEST(LayerTest, testLayerSetForwardError) {
    Layer *l;
    l = new Layer(10, 10);
    l->setForwardError(0.1);
    ASSERT_EQ(l->getForwardError(0), 0.1);
}

TEST(LayerTest, testLayerSetBackwardError) {
    Layer *l;
    l = new Layer(10, 10);
    l->setBackwardError(0.1);
    ASSERT_EQ(l->getBackwardError(0), 0.1);
}
