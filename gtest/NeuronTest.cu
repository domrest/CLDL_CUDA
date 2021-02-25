#include "cldl/Neuron.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>

using namespace std;

__global__ void test1(float *n){
    *n = *n * 4.0;
}
__global__ void test3(double *n){
//    n = new double[2];
    n[0] = 2.0;
}
__global__ void test2(Neuron *n){
//    n = new Neuron(1);
//    n->setGlobalError(2.0);
}

TEST(CUDATest, testfloat){
    float *a, *d_a;
    a = (float*)malloc(sizeof(float));
    *a = 2.0f;

    cudaMalloc((void**)&d_a,sizeof(float));
    cudaMemcpy(d_a, a, sizeof(float), cudaMemcpyHostToDevice);

    test1<<<1,1>>>(d_a);
    cudaMemcpy(a, d_a, sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(8.0f, *a);
}

TEST(CUDATest, testHelloWorld) {

    string hello = "Hello World";
    ASSERT_EQ(hello, "Hello World");

}

TEST(CUDATest, newDoubleList) {
    double *n, *d_n;

    cudaMalloc((void**)&d_n,sizeof(double)*2);

    test3<<<1,1>>>(d_n);

    n = new double[2];
    cudaMemcpy(n, d_n, sizeof(double)*2, cudaMemcpyDeviceToHost);
    ASSERT_EQ(n[0], 2.0);

}
TEST(NeuronTest, testNeuronInitializationCuda){
    Neuron *n;

    n = new Neuron(1);
    ASSERT_EQ(n->getNInputs(), 1);
}
//
//TEST(NeuronTest, testNeuronInitialization){
//    Neuron *n;
//
//    n = new Neuron(1);
//    n->setGlobalError(2.0);
//    ASSERT_EQ(n->getGlobalError(), 2.0);
//}

//int main(){
//
//
//}
