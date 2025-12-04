#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>

__global__ void convolution(const float* input, float* output ){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0;
    for(int i = 0; i < 3; ++i){
        result += input[tid + i];
    }
    output[tid] = result/3;
}

bool compare_array(int n, const float* a, const float* b){
    for(int i = 0; i < n; ++i){
        if (a[i] != b[i] ){
            printf("False at index: %d \n", i);
            return false;
        }
    }
    return true;
}

int main(){
    int num_elems = 1024;
    std::vector<float> in(num_elems +2, 2.0); 
    std::vector<float> out(num_elems, 0.0);

    float* d_inp;
    float* d_out;
    
    // allocate memory on gpu 
    cudaMalloc(&d_inp, sizeof(float) * (num_elems + 2));
    cudaMalloc(&d_out, sizeof(float) * (num_elems));

    //copy data from ram to gpu ram
    cudaMemcpy(d_inp, in.data(),sizeof(float) * (num_elems + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(),sizeof(float) * (num_elems), cudaMemcpyHostToDevice);

    int block_size = 128;
    int blocks = num_elems / block_size;

    // run convolution on gpu
    convolution<<<blocks, block_size>>>(d_inp, d_out);

    //get results back onto the ram
    cudaMemcpy(out.data(),d_out, sizeof(float) * (num_elems), cudaMemcpyDeviceToHost);

    std::vector<float> exp(num_elems, 2.0);
    bool ans = compare_array(num_elems, out.data(), exp.data());
    if(ans == true){
        printf("Correct results\n");
    }
    else{
        printf("wrong answers \n");
    }
    cudaFree(d_inp);
    cudaFree(d_out);
}