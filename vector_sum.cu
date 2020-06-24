#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <limits.h> 
#include <stdlib.h>
#include "common.cuh"

void initializeVector(int * vector, int VECTOR_QNT)
{
    for(size_t i=0; i<VECTOR_QNT; i++)
    {
        vector[i] = i;
    }
}

void CPUVectorSum(int * a, int * b, int * c, int VECTOR_QNT)
{
    for(size_t i=0; i<VECTOR_QNT; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void GPUVectorSum(int * a, int * b, int * c, int VECTOR_QNT) {
    int n = VECTOR_QNT;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

int checkCorrect(int * vector, int VECTOR_QNT)
{
    int errors = 0;

    for(size_t i=0; i<VECTOR_QNT; i++)
    {
        if(vector[i] != i*2)
        {
            errors++;
        }
    }

    return errors;
}

int main(int argc, char *argv[])
{
    if(argc!=2)
    {
        printf("Please input only the vectorsize!\n");
        exit(1);
    }

    char *p;
    int VECTOR_QNT;
    int VECTOR_SIZE;

    errno = 0;
    long conv = strtol(argv[1], &p, 10);

    // Check for errors: e.g., the string does not represent an integer
    // or the integer is larger than int
    if (errno != 0 || *p != '\0' || conv > INT_MAX) {
        // Put here the handling of the error, like exiting the program with
        // an error message
    } else {
        // No error
        VECTOR_QNT = conv;    
    }

    VECTOR_SIZE = VECTOR_QNT*sizeof(int);

    int * a_cpu = (int*)malloc(VECTOR_SIZE);
    int * b_cpu = (int*)malloc(VECTOR_SIZE);
    int * c_cpu = (int*)malloc(VECTOR_SIZE);

    int * a_gpu = NULL;
    int * b_gpu = NULL;
    int * c_gpu = NULL;

    clock_t start, end;
    dim3 grid, block;

    initializeVector(a_cpu, VECTOR_QNT);
    initializeVector(b_cpu, VECTOR_QNT);

    // Allocating memory for GPU DMAs
    cudaMalloc((int **)&a_gpu, VECTOR_SIZE);
    cudaMalloc((int **)&b_gpu, VECTOR_SIZE);
    cudaMalloc((int **)&c_gpu, VECTOR_SIZE);

    cudaMemcpy(a_gpu, a_cpu, VECTOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, VECTOR_SIZE, cudaMemcpyHostToDevice);

    start = clock();
    CPUVectorSum(a_cpu,b_cpu,c_cpu, VECTOR_QNT);
    end = clock();
    printf("Time=%f, Errors on CPU = %d\n", ((double) (end - start)) / CLOCKS_PER_SEC, checkCorrect(c_cpu, VECTOR_QNT));

    start = clock();
    GetGPUGridConfig(grid, block);
    GPUVectorSum<<<grid,block>>>(a_gpu,b_gpu,c_gpu, VECTOR_QNT);
    CheckGpuPanic();
    cudaDeviceSynchronize();
    end = clock();

    cudaMemcpy(c_cpu, c_gpu, VECTOR_SIZE, cudaMemcpyDeviceToHost);
    printf("Time=%f, Errors on GPU = %d\n", ((double) (end - start)) / CLOCKS_PER_SEC, checkCorrect(c_cpu, VECTOR_QNT));

    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return 0;
}