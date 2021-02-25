#include "cldl/Neuron.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>

using namespace std;


TEST(NeuronTest, testNeuronIntialisationAndNInputs){
    Neuron *n;

    n = new Neuron(1);
    ASSERT_EQ(n->getNInputs(), 1);
}


//int main(){
//
//
//}
