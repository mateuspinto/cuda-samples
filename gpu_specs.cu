#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void get_grid_config (dim3 &grid, dim3 &block)
{//Get the device properties

    static bool flag = 0;
    static dim3 lgrid, lthreads;

    if (!flag) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);

        //Adjust the grid dimensions based on the device properties
        int num_blocks = 1024 * 2 * devProp.multiProcessorCount;
        lgrid = dim3(num_blocks);
        lthreads = dim3(devProp.maxThreadsPerBlock / 4);
        flag = 1;
    }

    grid = lgrid;
    block = lthreads;
}

int main() {
    dim3 grid, block;
    get_grid_config(grid, block);
    printf("Recommended grid:(%d,%d,%d)\n", grid.x, grid.y, grid.z);
    printf("Recommended size:(%d,%d,%d)\n", block.x, block.y, block.y);

    return 1;
}