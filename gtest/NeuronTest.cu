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

TEST(Neuron, testSetInputs){
    Neuron *n;
    n = new Neuron(3);
    n->setInput(0,2.0);
    n->setInput(1,4.0);
    n->setInput(2,6.0);

    n->propInputs(1,3.0);

    ASSERT_EQ(n->getInput(0), 2.0);
    ASSERT_EQ(n->getInput(1), 3.0);
    ASSERT_EQ(n->getInput(2), 6.0);
}

TEST(NeuronTest, testSetInputErrors) {
    Neuron *n;
    n = new Neuron(4);
    n->setForwardError(2.0);
    n->propErrorForward(2, 4.0);
    ASSERT_EQ(n->getInputError(1), 2.0);
    ASSERT_EQ(n->getInputError(2), 4.0);

}


TEST(NeuronTest, testSumAndMaxMin){
    double *sum, *d_sum, *max, *d_max, *min, *d_min, *list, *d_list;

    gpu_allocateDouble(&d_sum, 0.0);
    gpu_allocateDouble(&d_max, 1.0);
    gpu_allocateDouble(&d_min, 1.0);


    cudaMalloc((void**)&d_list, sizeof(double)*4);

    list = new double[4];
    list[0] = 0.5;
    list[1] = 1.0;
    list[2] = 1.5;
    list[3] = 2.0;


    sum = (double*)malloc(sizeof(double));
    max = (double*)malloc(sizeof(double));
    min = (double*)malloc(sizeof(double));

    cudaMemcpy(d_list, list, sizeof(double)*4, cudaMemcpyHostToDevice);

    gpu_getSumAndMaxMin<<<1,1>>>(d_sum, d_max, d_min, d_list, 4);
    cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(min, d_min, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(*sum, 5.0);
    ASSERT_EQ(*max, 2.0);
    ASSERT_EQ(*min, 0.5);

}

TEST(NeuronTest, testDotProduct){
    double *d_list1, *d_list2, *list1, *list2, *d_value, *d_target, *target;

    gpu_allocateDouble(&d_target, 0.0);
    cudaMalloc((void**)&d_list1, sizeof(double)*4);
    cudaMalloc((void**)&d_list2, sizeof(double)*4);
    cudaMalloc((void**)&d_value, sizeof(double)*4);

    list1 = new double[4];
    list1[0] = 1.0;
    list1[1] = 1.0;
    list1[2] = 1.0;
    list1[3] = 1.0;

    list2 = new double[4];
    list2[0] = 1.0;
    list2[1] = 2.0;
    list2[2] = 3.0;
    list2[3] = 4.0;

    target = (double*)malloc(sizeof(double));

    cudaMemcpy(d_list1, list1, sizeof(double)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_list2, list2, sizeof(double)*4,cudaMemcpyHostToDevice);

    gpu_dotProduct<<<1,2>>>(d_list1, d_list2, d_value, d_target, 4);

    cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(*target, 10.0);
}

TEST(NeuronTest, testSetMidError){
    Neuron *n;
    n = new Neuron(4);
    n->setMidError(2.0);
    ASSERT_EQ(n->getInputMidErrors(1), 2.0);
    n->setMidError(3.0);
    ASSERT_EQ(n->getInputMidErrors(1), 3.0);
}

TEST(NeuronTest, testCalcMidError){
    Neuron *n;
    n = new Neuron(4);
    n->setMidError(2.0);
    n->calcMidError();
    ASSERT_EQ(n->getMidError(), 0.0);
}

int main(int argc, char** argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
