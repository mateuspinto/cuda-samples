#include <stdio.h>

#define VECTOR_QNT 1024

void initializeVector(int * vector)
{
    for(size_t i=0; i<VECTOR_QNT; i++)
    {
        vector[i] = i;
    }
}

void CPUVectorSum(int * a, int * b, int * c)
{
    for(size_t i=0; i<VECTOR_QNT; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void GPUVectorSum(int * a, int * b, int * c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int checkCorrect(int * vector)
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

int main()
{
    int * a = NULL;
    int * b = NULL;
    int * c_cpu = (int*)malloc(VECTOR_QNT*sizeof(int));
    int * c_gpu = NULL;

    // Allocating memory for GPU DMAs
    cudaMallocManaged(&a, VECTOR_QNT*sizeof(int));
    cudaMallocManaged(&b, VECTOR_QNT*sizeof(int));
    cudaMallocManaged(&c_gpu, VECTOR_QNT*sizeof(int));

    initializeVector(a);
    initializeVector(b);

    CPUVectorSum(a,b,c_cpu);
    GPUVectorSum<<<4,256>>>(a,b,c_gpu);
    cudaDeviceSynchronize();

    printf("Errors on CPU = %d\n", checkCorrect(c_cpu));
    printf("Errors on GPU = %d\n", checkCorrect(c_gpu));

    return 1;
}