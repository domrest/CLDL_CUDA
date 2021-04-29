#include "../include/cldl/Net.h"

#include <fstream>
#include <chrono>
#include <iostream>

using namespace std;


int* testNet(int layerCount, int nInputs) {
    int iterations = 2;
    Net *net;

    int *nNeuronsP = new int[layerCount];
    double *inputsp = new double[nInputs];

    for (int i = 0; i < layerCount; i++) {
        nNeuronsP[i] = layerCount - i;
    }
    for (int i = 0; i < nInputs; i++) {
        inputsp[i] = 1.0;
    }
    double leadError = 1;
    double learningRate = 1;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // init time
    net = new Net(layerCount, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);
    net->setErrorCoeff(0, 1, 0, 0, 0, 0);

    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    //prop inputs time
    for (int i = 0; i < iterations; i++) {
        net->setInputs(inputsp);
        net->propInputs();
    }
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

    int *num = new int[2];
    int num1 = (end1 - begin).count();
    int num2 = (end2 - end1).count();
    num[0] = num1;
    num[1] = num2;

    std::cout << "Time difference init = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin).count()
              << "[ns]" << std::endl;
    std::cout << "Time difference prop = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - end1).count()
              << "[ns]" << std::endl;


    delete net;
    return num;
}

int main()
{
    ofstream file1;
    ofstream file2;
    file1.open("C:\\Users\\domre\\CLionProjects\\CLDL_CUDA\\output1.csv", ios::trunc);
    file2.open("C:\\Users\\domre\\CLionProjects\\CLDL_CUDA\\ouptut2.csv", ios::trunc);
    for (int layers = 2; layers<50; layers++){
        for (int nNeurons = 2; nNeurons<50; nNeurons++){

            std::cout << "Layer count = " << layers << " input count = "<< nNeurons;
            std::cout << endl;
            int* num = testNet(layers, nNeurons);
            file1 << num[0] << ",";
            file2 << num[1] << ",";
            std::cout << endl;
        }
        file1 << "\n";
        file2 << "\n";
    }
    file1.close();
    file2.close();
    return 0;

}

