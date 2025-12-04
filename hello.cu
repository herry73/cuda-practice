#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu(){
    int tid = blockIdx.x*3 + threadIdx.x;
    printf("Hello from Thread:%d Block:%d \n", tid, blockIdx.x);
}

int main(){
    printf("Hello from CPU \n");
    hello_from_gpu<<<3,3>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU \n");
    return 0;
}