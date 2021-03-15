#include "cldl/Layer.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>

using namespace std;

__global__ void checkNInputs(Neuron* n){
    int i = threadIdx.x;
    *n[i].nInputs = 2;
}

TEST(CUDATest, testObjectPointerCalls){
    Neuron* n;
    n = (Neuron*) (malloc(sizeof(Neuron) * 5));
    for (int i=0; i<5; i++){
        Neuron* j = new Neuron(1);
        n[i] = *j;
    }

    Neuron* d_n;
    cudaMalloc((void**) &d_n, sizeof(Neuron)*5);
    cudaMemcpy(d_n, n, sizeof(Neuron)*5, cudaMemcpyHostToDevice);

    checkNInputs<<<1,5>>>(d_n);
    ASSERT_EQ(n[2].getNInputs(), 2);

}

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
