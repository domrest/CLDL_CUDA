#include "cldl/Neuron.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>

using namespace std;


TEST(NeuronTest, testNeuronIntialisationAndNInputs){
    Neuron *n;
    n = new Neuron(1);
    ASSERT_EQ(n->getNInputs(), 1);
}



TEST(NeuronTest, testSetLearningRate){
    Neuron *n;
    n = new Neuron(1);
    n->setLearningRate(2.0);
    ASSERT_EQ(n->getLearningRate(), 2.0);
}

TEST(NeuronTest, testSetInputErrors){
    Neuron *n;
    n = new Neuron(4);
    n->setForwardError(2.0);
    n->propErrorForward(2, 4.0);
    ASSERT_EQ(n->getInputError(1), 2.0);
    ASSERT_EQ(n->getInputError(2), 4.0);

}
//int main(){
//
//
//}
