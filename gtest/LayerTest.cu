#include "cldl/Layer.h"
#include "gtest/gtest.h"

using namespace std;

__global__ void checkNInputs(Neuron* n){
    int i = threadIdx.x;
    *n[i].nInputs = 2;
}

TEST(CUDATest, testObjectPointerCallsList){
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

//TODO testLayerDestructor

//TODO testInitLayer

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
    l = new Layer(100, 10);

    double in[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    l->setInputs(in);
    ASSERT_EQ(l->inputs[3], 4.0);

    Neuron *n;
    n = l->getNeuron(3);
    ASSERT_EQ(n->getNInputs(), 10);
    ASSERT_EQ(n->getInput(5), 6.0);
    ASSERT_EQ(n->getInput(2), 3.0);

    Neuron *n2;
    n2 = l->getNeuron(99);
    ASSERT_EQ(n2->getNInputs(), 10);
    ASSERT_EQ(n2->getInput(5), 6.0);
    ASSERT_EQ(n2->getInput(2), 3.0);
}

//TODO testLayerPropInputs
TEST(LayerTest, testLayerPropInputs) {
    Layer *l;
    l = new Layer(200, 10);

    double prevLayerOuts[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double* gpu_InputOutputs;
    cudaMalloc( (void**) &gpu_InputOutputs, sizeof(double)*10);
    cudaMemcpy(gpu_InputOutputs, prevLayerOuts, sizeof(double)*10,cudaMemcpyHostToDevice);

    l->propInputs(gpu_InputOutputs);
    //ASSERT_EQ(l->inputs[3], 4.0);
    Neuron *n;
    n = l->getNeuron(3);
    ASSERT_EQ(n->getNInputs(), 10);
    ASSERT_EQ(n->getInput(5), 6.0);
    ASSERT_EQ(n->getInput(2), 3.0);

    Neuron *n2;
    n2 = l->getNeuron(150);
    ASSERT_EQ(n2->getNInputs(), 10);
    ASSERT_EQ(n2->getInput(5), 6.0);
    ASSERT_EQ(n2->getInput(2), 3.0);
}

/*TEST(LayerTest, testLayerSetForwardError) {
    Layer *l;
    l = new Layer(10, 10);
    l->setForwardError(0.1);
    ASSERT_EQ(l->getForwardError(0), 0.1);
}*/

TEST(LayerTest, testLayerSetBackwardError) {
    Layer *l;
    l = new Layer(10, 10);
    l->setBackwardError(0.01);
    ASSERT_EQ(l->leadBackwardError, 0.01);
    ASSERT_EQ(l->getBackwardError(5), 0.0025);
}

TEST(LayerTest, testLayerSetErrorCoeff) {
    Layer *l;
    l = new Layer(10, 10);
    l->setErrorCoeff(0, 1, 0, 0, 0, 0);

    Neuron *n;
    n = l->getNeuron(5);
    ASSERT_EQ(n->getBackwardsCoeff(), 1.0);
}

TEST(LayerTest, testLayerUpdateWeights) {
    Layer *l;
    l = new Layer(10, 10);
    l->setBackwardError(2.0);
    l->setlearningRate(2.0);
    l->setErrorCoeff(0, 1, 0, 0, 0, 0);
    double in[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    l->setInputs(in);
    l->updateWeights();

    Neuron *n;
    n = l->getNeuron(5);
    ASSERT_EQ(n->getWeight(0), 1.0);
    ASSERT_EQ(n->getWeight(1), 2.0);
    /* This test doesn't pass if BackwardError
       and LearningRate are set to 0.1.
       It says:
       "getWeight(0) = 0.0025."
       "This does not equal 0.0025" */
}

//TODO testLayerCalcErrorWeightProductSum
TEST(LayerTest, testLayerCalcErrorWeightProductSum) {
    Layer *l;
    l = new Layer(10, 12);
    l->setBackwardError(2.0);
    l->setlearningRate(2.0);
    double in[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    l->setInputs(in);
    l->updateWeights();
    double* sumlist;
    sumlist = l->calcErrorWeightProductSum();

    Neuron *n;
    n = l->getNeuron(2);
    ASSERT_EQ(l->leadBackwardError, 2.0);
    ASSERT_EQ(l->getBackwardError(0), 0.5);
    ASSERT_EQ(n->getWeight(11), 12.0);
    ASSERT_EQ(n->getInput(11), 12.0);

    ASSERT_EQ(n->getErrorWeightProducts(0),0.5);
    ASSERT_EQ(n->getErrorWeightProducts(1),1.0);
    ASSERT_EQ(n->getErrorWeightProducts(11), 6.0);

    ASSERT_EQ(l->getSum(0), 5);
    ASSERT_EQ(l->getSum(1), 10);
    ASSERT_EQ(l->getSum(11), 60);
}

//TODO test_propErrorBackward
TEST(LayerTest, testLayerpropErrorBackwards) {
    //Create "final" layer
    Layer *l;
    l = new Layer(10, 12);
    l->setlearningRate(2.0);
    l->setBackwardError(2.0);
    double in[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    l->setInputs(in);
    l->updateWeights();
    double *sumlist;
    sumlist = l->calcErrorWeightProductSum();

    //Create "previous" layer
    Layer *l2;
    l2 = new Layer(12, 10);
    l2->setBackwardError(2.0);
    l2->setlearningRate(2.0);
    l2->propErrorBackward(sumlist);

    ASSERT_EQ(l2->getnNeurons(), 12);
    ASSERT_EQ(l2->getBackwardError(0), 1.25);
    ASSERT_EQ(l2->getBackwardError(1), 2.5);
    ASSERT_EQ(l2->getBackwardError(11), 15);
}

//TODO testLayerCalcOutputs
