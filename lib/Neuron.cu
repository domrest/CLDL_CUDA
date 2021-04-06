#include "cldl/Neuron.h"

#include <cuda_runtime.h>



//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

__host__ Neuron::Neuron(int _nInputs)
{
    // initialisation
    gpu_allocateInt(&nInputs, _nInputs);
    gpu_allocateInt(&myLayerIndex, 0);
    gpu_allocateInt(&myNeuronIndex, 0);
    cudaMalloc((void**)&initialWeights, sizeof(double)*_nInputs);
    gpu_allocateDouble(&learningRate, 0.0);

    gpu_allocateInt(&iHaveReported, 0);

    // forward propagation of inputs
    cudaMalloc((void**)&inputs, sizeof(double)*_nInputs);
    gpu_allocateDouble(&bias, 0.0);
    gpu_allocateDouble(&sum, 0.0);
    gpu_allocateDouble(&output, 0.0);

    // forward propagation of error
    cudaMalloc((void**)&inputErrors, sizeof(double)*_nInputs);
    gpu_allocateDouble(&forwardError, 0.0);

    // back propagation of error
    gpu_allocateDouble(&backwardError, 0.0);
    cudaMalloc((void**)&ErrorWeightProducts, sizeof(double)*_nInputs);

    // mid propagation of error
    cudaMalloc((void**)&inputMidErrors, sizeof(double)*_nInputs);
    gpu_allocateDouble(&midError, 0.0);


    //
    // learning variables
    //
    gpu_allocateDouble(&backwardsCoeff, 0.0);
    gpu_allocateDouble(&midCoeff, 0.0);
    gpu_allocateDouble(&forwardCoeff, 0.0);
    gpu_allocateDouble(&globalCoeff, 0.0);

    cudaMalloc((void**)&weights, sizeof(double)*_nInputs);

    gpu_allocateDouble(&weightSum, 0.0);
    gpu_allocateDouble(&maxWeight, 1.0);
    gpu_allocateDouble(&minWeight, 1.0);
    gpu_allocateDouble(&weightChange, 0.0);
    gpu_allocateDouble(&weightsDifference, 0.0);
    gpu_allocateInt(&actMet, 0);

    // global setting
    gpu_allocateDouble(&globalError, 0.0);
    gpu_allocateDouble(&localError, 0.0);
    gpu_allocateDouble(&echoCoeff, 0.0);
    gpu_allocateDouble(&localCoeff, 0.0);

    gpu_allocateDouble(&overallError, 0.0);
    gpu_allocateDouble(&echoError, 0.0);
    cudaMalloc((void**)&echoErrors, sizeof(double)*_nInputs);

    //cout << "neuron" << endl;

}

__host__ Neuron::~Neuron(){
    //initialisation
    cudaFree(nInputs);
    cudaFree(learningRate);
    cudaFree(myLayerIndex);
    cudaFree(initialWeights);
    cudaFree(myNeuronIndex);

    cudaFree(iHaveReported);

    // forward propagation of inputs
    cudaFree(inputs);
    cudaFree(bias);
    cudaFree(sum);
    cudaFree(output);

    // forward propagation of error
    cudaFree(inputErrors);
    cudaFree(forwardError);

    // back propagation of error
    cudaFree(backwardError);
    cudaFree(ErrorWeightProducts);

    // mid propagation of error
    cudaFree(inputMidErrors);
    cudaFree(midError);


    //learning
    cudaFree(backwardsCoeff);
    cudaFree(midCoeff);
    cudaFree(forwardCoeff);
    cudaFree(globalCoeff);
    cudaFree(weights);
    cudaFree(weightSum);
    cudaFree(maxWeight);
    cudaFree(minWeight);
    cudaFree(weightChange);
    cudaFree(weightsDifference);
    cudaFree(actMet);

    // global setting
    cudaFree(globalError);
    cudaFree(localError);
    cudaFree(echoCoeff);
    cudaFree(localCoeff);

    cudaFree(overallError);
    cudaFree(echoError);
    cudaFree(echoErrors);
}


//*************************************************************************************
//initialisation:
//*************************************************************************************

//TODO test init neuron
__host__ void Neuron::initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am){
    cudaMemcpy(myLayerIndex, &_layerIndex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(myNeuronIndex, &_neuronIndex, sizeof(int), cudaMemcpyHostToDevice);
    switch(_wim) {
        case W_ZEROS:
            gpu_setValuesInArray<<<1,getNInputs()>>>(0, weights);
            break;
        case W_ONES:
            gpu_setValuesInArray<<<1,getNInputs()>>>(1, weights);
            break;
        case W_RANDOM:
            //TODO set the random
//            weights[i] = (((double) rand() / (RAND_MAX))); //* 2) -1;
            break;
            //cout << " Neuron: weight is: " << weights[i] << endl;
            /* rand function generates a random function between
             * 0 and RAND_MAX, after the devision the weights are
             * set to a value between 0 and 1 */
    }
    cudaMemcpy(initialWeights, weights, sizeof(double)*getNInputs(), cudaMemcpyDeviceToDevice);

    gpu_setDouble<<<1,1>>>(weightSum, 0);
    gpu_getSumAndMaxMin<<<1,1>>>(weightSum, maxWeight, minWeight, weights, getNInputs());

    switch (_bim){
        case B_NONE:
            gpu_setDouble<<<1,1>>>(bias, 0.0);
            break;
        case B_RANDOM:
            gpu_setDouble<<<1,1>>>(bias, ((double)rand()/RAND_MAX));
            break;
    }
    switch(_am){
        case Act_Sigmoid:
            gpu_setInt<<<1,1>>>(actMet, 0);
            break;
        case Act_Tanh:
            gpu_setInt<<<1,1>>>(actMet, 1);
            break;
        case Act_NONE:
            gpu_setInt<<<1,1>>>(actMet, 2);
            break;
    }
}

__host__ void Neuron::setLearningRate(double _learningRate){
    gpu_setDouble<<<1,1>>>(learningRate, _learningRate);
}

__device__ void device_setLearningRate(Neuron* n, double _learningRate){
    *n->learningRate = _learningRate;
}

__host__ double Neuron::getLearningRate() {
    double _learningRate;
    cudaMemcpy(&_learningRate, learningRate, sizeof(double), cudaMemcpyDeviceToHost);
    return _learningRate;
}


//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************
__host__ void Neuron::setInput(int _index, double _value) {
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputs);
}

__host__ double Neuron::getInput(int index) {
    double _input = 0.0;
    assert(index < getNInputs());

    double* input = inputs + index;
    cudaMemcpy(&_input, input, sizeof(double), cudaMemcpyDeviceToHost);
    return _input;
}

__host__ void Neuron::propInputs(int _index,  double _value){
    assert((_index>=0)&&(_index < getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value,_index, inputs);
}


__device__ void device_calcOutput(Neuron* n){
    double* _value = new double[1024];
    int nInputs = *(n->nInputs);
    device_dotProduct((*n).inputs, (*n).weights, _value, (*n).sum, nInputs);
}

__device__ void device_calcOutputCont(Neuron* n, int* _layerHasReported){
    if (*(*n).myLayerIndex == 0){
        *(*n).sum = *(*n).sum * 0.01;
    }
    *(*n).sum += *(*n).bias;
    device_doActivation((*n).output, (*n).sum, (*n).actMet);
    *(*n).iHaveReported = *_layerHasReported;
    if (*(*n).output > 0.49 && *(*n).iHaveReported == 0){
        *(*n).iHaveReported = 1;
    }
    *_layerHasReported = *(*n).iHaveReported;
}

//int Neuron::calcOutput(int _layerHasReported){
//    sum=0;
//    for (int i=0; i<nInputs; i++){
//        sum += inputs[i] * weights[i];
//    }
//    sum += bias;
//    if (myLayerIndex == 0){
//        sum = sum * 0.01;
//    }
//    assert(std::isfinite(sum));
//    output = doActivation(sum);
//    iHaveReported = _layerHasReported;
//    if (output > 0.49 && iHaveReported == 0){
//        //cout << "I'm saturating, " << output << " layer: " << myLayerIndex << " neuron: " << myNeuronIndex << endl;
//        iHaveReported = 1;
//    }
//    assert(std::isfinite(output));
//    return iHaveReported;
//}


//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

__host__ void Neuron::setForwardError(double _value) {
    gpu_setValuesInArray<<<1, getNInputs()>>>(_value, inputErrors);
}

__host__ double Neuron::getInputError(int _index) {
    double _inputError = 0.0;
    assert(_index < getNInputs());

    double* inputError = inputErrors + _index;
    cudaMemcpy(&_inputError, inputError, sizeof(double), cudaMemcpyDeviceToHost);
    return _inputError;
}

__host__ void Neuron::propErrorForward(int _index, double _value){
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputErrors);
}


//TODO calcForwardError
__device__ void calcForwardError(Neuron* n){
    double* _value = new double[1024];
    device_dotProduct((*n).inputErrors,(*n).weights,_value, (*n).calcForwardOutput,*(*n).nInputs);
    device_doActivationPrime((*n).forwardError, (*n).sum, (*n).actMet);
    *(*n).forwardError = *(*n).forwardError * *(*n).calcForwardOutput;
}

//__host__ void Neuron::calcForwardError() {
//    double* _value;
//    cudaMalloc((void**)&_value, sizeof(double)*getNInputs());
//    gpu_dotProduct<<<1, getNInputs()>>>(inputErrors, weights, _value, fowardError, getNInputs());

    //TODO forwardError must be multiplied with doActivationPrime(sum)
    //TODO assert forwardError isFinite
//}

__host__ double Neuron::getForwardError() {
    double _forwardError = 0.0;
    cudaMemcpy(&_forwardError, forwardError, sizeof(double), cudaMemcpyDeviceToHost);
    return _forwardError;
}


//*************************************************************************************
//back propagation of error
//*************************************************************************************

//TODO Fix setBackwardError
__host__ void Neuron::setBackwardError(double _leadError){
    gpu_doActivationPrime<<<1,1>>>(backwardError, sum, actMet);
    gpu_multiplication<<<1,1>>>(_leadError,backwardError);
}

__device__ void device_setBackwardError(double _leadError, Neuron* n){
    device_doActivationPrime((*n).backwardError, (*n).sum, (*n).actMet);
    *(*n).backwardError = *(*n).backwardError * _leadError;
}

__global__ void gpu_setBackwardError(double _leadError, Neuron* n){
    double leadError = _leadError;
    device_setBackwardError(leadError, n);
}

__device__ void device_propErrorBackward(double _nextSum, Neuron* n){
    device_doActivationPrime((*n).backwardError, (*n).sum, (*n).actMet);
    *(*n).backwardError = *(*n).backwardError * _nextSum;
}

__global__ void gpu_propErrorBackward(double _nextSum, Neuron* n){
    double nextSum = _nextSum;
    device_propErrorBackward(nextSum, n);
}

__host__ double Neuron::getBackwardError(){
    double _backwardError = 0.0;
    cudaMemcpy(&_backwardError, backwardError, sizeof(double), cudaMemcpyDeviceToHost);
    return _backwardError;
}

__host__ double Neuron::getErrorWeightProducts(int index) {
    double _ewProd = 0.0;

    double* ewProd = ErrorWeightProducts + index;
    cudaMemcpy(&_ewProd, ewProd, sizeof(double), cudaMemcpyDeviceToHost);
    return _ewProd;
}

__host__ double Neuron::getEchoError() {
    double _echoError = 0;
    cudaMemcpy(&_echoError, echoError, sizeof(double), cudaMemcpyDeviceToHost);
    return _echoError;
}

__device__ void echoErrorBackward(double _nextSum, Neuron* n) {
    device_doActivationPrime((*n).echoError,(*n).sum,(*n).actMet);
    *(*n).echoError = *(*n).echoError * _nextSum;
}

__global__ void gpu_echoErrorBackward(double _nextSum, Neuron* n){
    double nextSum = _nextSum;
    echoErrorBackward(nextSum, n);
}
//*************************************************************************************
//MID propagation of error
//*************************************************************************************

__host__ void Neuron::setMidError(double _leadMidError) {
    gpu_setValuesInArray<<<1, getNInputs()>>>(_leadMidError, inputMidErrors);
}

__host__ double Neuron::getInputMidErrors(int index) {
    double _inputMidError = 0.0;
    assert(index < getNInputs());

    double* inputMidError = inputMidErrors + index;
    cudaMemcpy(&_inputMidError, inputMidError, sizeof(double), cudaMemcpyDeviceToHost);
    return _inputMidError;
}

//TODO find how to do midError = midError * doActivationPrime(sum)
__host__ void Neuron::calcMidError() {
    double* _value;
    cudaMalloc((void**)&_value, sizeof(double)*getNInputs());
    gpu_dotProduct<<<1, getNInputs()>>>(inputMidErrors, weights, _value, midError, getNInputs());
    double output = getMidError();
    gpu_doActivationPrime<<<1,1>>>(midError, sum, actMet);
    gpu_multiplication<<<1, 1>>>(output, midError);

}


__host__ double Neuron::getMidError() {
    double _midError = 0.0;
    cudaMemcpy(&_midError, midError, sizeof(double), cudaMemcpyDeviceToHost);
    return _midError;
}

__host__ void Neuron::propMidErrorForward(int _index, double _value){
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputMidErrors);
}




__host__ void Neuron::propMidErrorBackward(double _nextSum){
    //TODO needs test
    gpu_propError<<<1,1>>>(_nextSum, sum, actMet, midError);
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

//TODO getError

//*************************************************************************************
//learning
//*************************************************************************************

//TODO setErrorCoeff

__host__ double Neuron::getBackwardsCoeff(){
    double _backwardsCoeff = 0.0;
    cudaMemcpy(&_backwardsCoeff, backwardsCoeff, sizeof(double), cudaMemcpyDeviceToHost);
    return _backwardsCoeff;
}

//TODO updateWeights

__host__ double Neuron::getWeight(int index) {
    double _weight = 0.0;

    double* weight = weights + index;
    cudaMemcpy(&_weight, weight, sizeof(double), cudaMemcpyDeviceToHost);
    return _weight;
}

//*************************************************************************************
//global settings
//*************************************************************************************

//TODO setGlobalError

//TODO getGlobalError

//TODO setEchoError

//TODO echoErrorForward

//TODO calcEchoError

//*************************************************************************************
//local backpropagation of error
//*************************************************************************************

//TODO setLocalError

//TODO propGlobalErrorBackwardLocally

//TODO getLocalError

//*************************************************************************************
// getters:
//*************************************************************************************

__host__ double Neuron::getOutput(){
    double _output=0.0;
    cudaMemcpy(&_output, output, sizeof(double), cudaMemcpyDeviceToHost);
    return _output;
}

__host__ double Neuron::getSumOutput(){
    double _sum=0;
    cudaMemcpy(&_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);
    return _sum;
}

__host__ double Neuron::getMaxWeight(){
    double _maxWeight=0;
    cudaMemcpy(&_maxWeight, maxWeight, sizeof(double), cudaMemcpyDeviceToHost);
    return _maxWeight;
}

__host__ double Neuron::getMinWeight(){
    double _minWeight=0;
    cudaMemcpy(&_minWeight, minWeight, sizeof(double), cudaMemcpyDeviceToHost);
    return _minWeight;
}

__host__ double Neuron::getSumWeight(){
    double _weightSum=0;
    cudaMemcpy(&_weightSum, weightSum, sizeof(double), cudaMemcpyDeviceToHost);
    return _weightSum;
}


//double Neuron::getWeightChange(){
//    weightsDifference = 0;
//    weightChange = 0;
//    for (int i=0; i<nInputs; i++){
//        weightsDifference = weights[i] - initialWeights[i];
//        weightChange += pow(weightsDifference,2);
//    }
//    return (weightChange);
//}

//TODO getWeightDistance

__host__ int Neuron::getNInputs(){
    int _nInputs=0;
    cudaMemcpy(&_nInputs, nInputs, sizeof(int), cudaMemcpyDeviceToHost);
    return _nInputs;
}


//TODO getWeights

//TODO getInitWeights

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

//TODO saveWeights

//TODO printNeuron

//*************************************************************************************
//helper host functions:
//*************************************************************************************
__host__ void gpu_allocateInt(int** pointer, int value){
    cudaMalloc(pointer, sizeof(int));
    gpu_setInt<<<1,1>>>(*pointer, value);
}
__host__ void gpu_allocateDouble(double** pointer, double value){
    cudaMalloc(pointer, sizeof(double));
    gpu_setDouble<<<1,1>>>(*pointer, value);
}

//*************************************************************************************
//device CUDA kernels:
//*************************************************************************************
__device__ void device_propError(double _value, double* sum, int* actMet, double* errorLocation){
    double output = 0;
    device_doActivationPrime(&output, sum, actMet);
    *errorLocation = _value * output;
}

__device__ void device_doActivation(double* output, double* sum, int* actMet) {
    switch(*actMet){
        case 0:
            *output = (1/(1+(exp(-*sum)))) - 0.5;
            break;
        case 1:
            *output = tanh(*sum);
            break;
        case 2:
            *output = *sum;
            break;
    }
}

__device__ void device_doActivationPrime(double* output, double* input, int* actMet){
    switch(*actMet){
        case 0:
            device_doActivation(output, input, actMet);
            *output = 1 * (0.5 + *output) * (0.5 - *output); //exp(-_input) / pow((exp(-_input) + 1),2);
            break;
        case 1:
            *output = 1 - pow(tanh(*input), 2.0);
            break;
        case 2:
            *output = 1;
            break;
    }
}

//*************************************************************************************
//global CUDA kernels:
//*************************************************************************************
__global__ void gpu_propError(double _value, double* sum, int* actMet, double* errorLocation) {
    device_propError(_value, sum, actMet, errorLocation);
}


__global__ void gpu_setValuesInArray(double _value, double* list){
    list[threadIdx.x] = _value;
}

__global__ void gpu_setValueInArray(double _value, int index, double* list){
    list[index] = _value;
}

__global__ void gpu_getSumAndMaxMin(double* sum, double* max_list, double* list_min, double* list, int length){
    for (int i=0; i<length; i++){
        *sum = *sum + fabs(list[i]);
        *max_list = max(*max_list, list[i]);
        *list_min = min(*list_min, list[i]);
    }
}


__global__ void gpu_setInt(int* pointer, int value) {
    *pointer = value;
}

__global__ void gpu_setDouble(double* pointer, double value){
    *pointer = value;
}

__global__ void gpu_doActivation(double* output, double* sum, int* actMet) {
    device_doActivation(output, sum, actMet);
}

__global__ void gpu_doActivationPrime(double* output, double* input, int* actMet) {
    device_doActivationPrime(output, input, actMet);
}

__global__ void gpu_dotProduct(double* list1, double* list2, double* _value, double* _target, int arrayLength){
    int idx = threadIdx.x;
    int stride = blockDim.x;

    double target = 0.0;
    for (int i = idx; i < arrayLength; i+=stride){
        target += list1[i]*list2[i];
    }

    _value[idx] = target;
    __syncthreads();

    for (int size = stride/2; size>0; size/=2){
        if (idx < size){
            _value[idx] += _value[idx+size];
        }
        __syncthreads();
    }
    if (idx == 0){
        *_target = _value[0];
    }
}

__device__ void device_dotProduct(double* list1, double* list2, double* _value, double* _target, int arrayLength){
    int idx = threadIdx.x;
    int stride = 1;

    double target = 0.0;
    for (int i = idx; i < arrayLength; i+=stride){
        target += list1[i]*list2[i];
    }

    _value[idx] = target;
    __syncthreads();

    for (int size = stride/2; size>0; size/=2){
        if (idx < size){
            _value[idx] += _value[idx+size];
        }
        __syncthreads();
    }
    if (idx == 0){
        *_target = _value[0];
    }
}

__global__ void gpu_multiplication(double value, double* output){
    *output = value * *output;
}
