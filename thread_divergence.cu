// Tiny example to demonstrate thread divergence within a warp

#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>

__global__ void kernel(float *x, int n)
{
    assert(n == 32);

if (threadIdx.x < 16)
{
        x[threadIdx.x] = x[threadIdx.x] * 2.0f;
}
else
{
        x[threadIdx.x] = x[threadIdx.x] + 1.0f;
}
x[threadIdx.x] += 1.0f;


}

int main()
{
    const int N = 32;
    float h_x[N], *d_x;

    // Initialize host array
    for (int i = 0; i < N; i++)
    {
        h_x[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with one block of 32 threads
    kernel<<<1, N>>>(d_x, N);

    // Copy result back to host
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++)
    {
        printf("h_x[%d] = %f\n", i, h_x[i]);
    }

    // Free device memory
    cudaFree(d_x);

    return 0;
}
