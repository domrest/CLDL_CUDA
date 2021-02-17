#include "gtest/gtest.h"

__global__ void test1(float *n){
    *n = *n * 4.0;
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

//int main(){
//
//
//}
