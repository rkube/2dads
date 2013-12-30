/*
 *
 * CUDA specific initialization function
 *
 */

#include <iostream>
#include <vector>
#include "include/cuda_types.h"
#include "include/cuda_array2.h"


__global__ 
void d_init_sine(cuda::real_t* array, cuda::slab_layout layout, double* params) 
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col < layout.My) && (row < layout.Nx))
    {
        //printf("idx: %d, col: %d, row: %d, x = %f, y = %f\t\tkx = %f, ky = %f\n", idx, col, row, x, y, params[0], params[1]);
        array[idx] = sin(params[0] * cuda::PI * x) + sin(params[1] * cuda::PI * y);
    }
}


void init_simple_sine(cuda_array<cuda::real_t>* arr, 
        vector<double> initc,
        const double delta_x,
        const double delta_y,
        const double x_left,
        const double y_lo)
{
    cout << "init_simple_sine()\n";
    cuda::slab_layout layout = {x_left, delta_x, y_lo, delta_y, arr -> get_nx(), arr -> get_my()};

    dim3 grid = arr -> get_grid();
    dim3 block = arr -> get_block();

    double* params = initc.data();
    // Copy the parameters for the function to the device
    double* d_params;
    gpuErrchk(cudaMalloc( (double**) &d_params, initc.size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_params, params, sizeof(double) * initc.size(), cudaMemcpyHostToDevice));

    d_init_sine<<<grid, block>>>(arr -> get_array_d(0), layout, d_params);
    //d_init_sine<<<1, 1>>>(arr -> get_array_d(), layout, d_params);
    cudaDeviceSynchronize();
}







