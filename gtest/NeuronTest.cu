#include "cldl/Neuron.h"
#include "gtest/gtest.h"

using namespace std;


TEST(CUDATest, testCudaMalloc){
    double *d_a;
    double a = 2.0;
    cudaMallocManaged(&d_a, sizeof(double));
    gpu_setDouble<<<1,1>>>(d_a, 1.0);
    cudaMemcpy(&a, d_a, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(a, 1.0);

}

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

TEST(NeuronTest, TestDoActivation){
    double *d_result;
    double result = 0;
    int *actMet;
    gpu_allocateDouble(&d_result, 0.0);
    gpu_allocateInt(&actMet, 1);
    gpu_doActivation<<<1,1>>>(d_result, 2, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.9640275800758169);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setInt<<<1,1>>>(actMet, 2);
    gpu_doActivation<<<1,1>>>(d_result, 2, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 2.0);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setInt<<<1,1>>>(actMet, 0);
    gpu_doActivation<<<1,1>>>(d_result, 2, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.38079707797788231);

}

TEST(NeuronTest, TestDoActivationPrime){
    double *d_result;
    double result = 0;
    int *actMet;
    gpu_allocateDouble(&d_result, 0.0);
    gpu_allocateInt(&actMet, 0);
    gpu_doActivationPrime<<<1,1>>>(d_result, 2, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.10499358540350662);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setInt<<<1,1>>>(actMet, 1);
    gpu_doActivationPrime<<<1,1>>>(d_result, 2, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.070650824853164429);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setInt<<<1,1>>>(actMet, 2);
    gpu_doActivationPrime<<<1,1>>>(d_result, 2, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 1.0);

}

int main(int argc, char** argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
