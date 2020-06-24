#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello, World from GPU!\n");
}

void helloFromCPU()
{
    printf("Hello, World from CPU!\n");
}

int main()
{
    helloFromCPU();
    helloFromGPU<<<1,1>>>();
    cudaDeviceSynchronize();
    helloFromCPU();
}