#include "cldl/Layer.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>

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


TEST(LayerTest, testLayerSetLearningRate) {
    Layer *l;
    l = new Layer(10, 10);
    l->setlearningRate(0.1);

    Neuron *n;
    n = l->getNeuron(0);
    ASSERT_EQ(n->getLearningRate(), 0.1);
}


