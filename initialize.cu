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
void d_init_sine(cuda::real_t* array, cuda::slab_layout_t layout, double* params) 
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.Nx) || (row >= layout.My))
        return;
    array[idx] = sin(params[0] * cuda::PI * x) + sin(params[1] * cuda::PI * y);
}


__global__
void d_init_exp(cuda::real_t* array, cuda::slab_layout_t layout, double* params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.My) || (row >= layout.Nx))
        return;

    array[idx] = params[0] + params[1] * exp( -(x - params[2]) * (x - params[2]) / (2.0 * params[3] * params[3]) 
                                              -(y - params[4]) * (y - params[4]) / (2.0 * params[5] * params[5]));
}


__global__
void d_init_exp_log(cuda::real_t* array, cuda::slab_layout_t layout, double* params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.My) || (row >= layout.Nx))
        return;

    array[idx] = params[0] + params[1] * exp( -(x - params[2]) * (x - params[2]) / (2.0 * params[3] * params[3]) 
                                              -(y - params[4]) * (y - params[4]) / (2.0 * params[5] * params[5]));
    array[idx] = log(array[idx]);
}


__global__
void d_init_lapl(cuda::real_t* array, cuda::slab_layout_t layout, double* params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.My) || (row >= layout.Nx))
        return;

    array[idx] = exp(- 0.5 * (x * x + y * y)/(params[3] * params[3])) / (params[3] * params[3]) * 
                 ((x * x + y * y)/(params[3] * params[3]) - 2.0);
}


__global__
void d_init_mode_exp(cuda::cmplx_t* array, cuda::slab_layout_t layout, double* params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    if ((col >= layout.My) || (row >= layout.Nx))
        return;
    double n = double(row);
    double m = double(col);
    double amplitude = params[0];
    double modex = params[1];
    double modey = params[2];
    double damp = exp ( -((n-modex)*(n-modex) / 0.1) - ((m-modey)*(m-modey) / 0.1) ); 
    double phase = 0.56051 * 2.0 * cuda::PI;  

    array[idx] = make_cuDoubleComplex(damp * amplitude * cos(phase), damp * amplitude * sin(phase));
}


//__global__
//void d_init_mode(cuda::cmplx_t* array, cuda::slab_layout_t layout, double* params)
//{
//    const uint idx = uint(params[2]) * layout.My + uint(params[1]);
//    double phase = 0.56051 * 2.0 * cuda::PI;
//    array[idx] = make_cuDoubleComplex(params[0] * sin(phase), params[0] * cos(phase));
//    printf("d_init_mide: mode (%d, %d) at idx= %d = (%f, %f)\n",
//            uint(params[2]), uint(params[1]), idx, params[0] * sin(phase), params[0] * cos(phase));
//}


__global__
void d_init_mode(cuda::cmplx_t* array, cuda::slab_layout_t layout, double amp, uint col, uint row)
{
    const uint idx = row * layout.My + col;
    const double phase = 0.56051 * cuda::TWOPI;
    array[idx] = make_cuDoubleComplex(amp * cos(phase), amp * sin(phase));
    printf("d_init_mode: mode(%d, %d) at idx = %d = (%f, %f)\n",
            row, col, idx, cos(phase), sin(phase));
}

void init_simple_sine(cuda_array<cuda::real_t>* arr, 
        vector<double> initc,
        const double delta_x,
        const double delta_y,
        const double x_left,
        const double y_lo)
{
    cout << "init_simple_sine()\n";
    cuda::slab_layout_t layout = {x_left, delta_x, y_lo, delta_y, arr -> get_nx(), arr -> get_my()};

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


void init_gaussian(cuda_array<cuda::real_t>* arr,
        vector<double> initc,
        const double delta_x,
        const double delta_y,
        const double x_left,
        const double y_lo,
        bool log_theta)
{
    cuda::slab_layout_t layout = {x_left, delta_x, y_lo, delta_y, arr -> get_nx(), arr -> get_my()};

    double* params = initc.data();
    double* d_params;

    gpuErrchk(cudaMalloc( (double**) &d_params, initc.size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_params, params, sizeof(double) * initc.size(), cudaMemcpyHostToDevice));

    if (log_theta)
    {
        cout << "Initializing logarithmic theta\n";
        d_init_exp_log<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, d_params);
    }
    else
    {
        cout << "Initializing theta\n";
        d_init_exp<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, d_params);
    }
    cout << "initc = (" << params[0] << ", " << params[1] << ", " << params[2] << ", ";
    cout << params[3] << ", " << params[4] << ", " << params[5] << ")\n";
    cudaDeviceSynchronize();
}


void init_invlapl(cuda_array<cuda::real_t>* arr,
        vector<double> initc,
        const double delta_x,
        const double delta_y,
        const double x_left,
        const double y_lo)
{
    cout << "init_invlapl\n";
    cuda::slab_layout_t layout = {x_left, delta_x, y_lo, delta_y, arr -> get_nx(), arr -> get_my()};

    double* params = initc.data();
    double* d_params;
    cout << "initc = (" << params[0] << ", " << params[1] << ", " << params[2] << ", ";
    cout << params[3] << ", " << params[4] << ", " << params[5] << ")\n";

    gpuErrchk(cudaMalloc( (double**) &d_params, initc.size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_params, params, sizeof(double) * initc.size(), cudaMemcpyHostToDevice));

    d_init_lapl<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, d_params);
    cudaDeviceSynchronize();
}


void init_mode(cuda_array<cuda::cmplx_t>* arr,
        vector<double> initc,
        const double delta_x,
        const double delta_y,
        const double x_left,
        const double y_lo)
{
    // We call arr -> get_my() which is alreade reduced to My/2+1 from the slab since arr is of
    // type cuda_array<cuda::cmplx_t>
    cuda::slab_layout_t layout = {x_left, delta_x, y_lo, delta_y, arr -> get_nx(), arr -> get_my()};

    //double* params = initc.data();
    const unsigned int num_modes = initc.size() / 3;
    //for(uint i = 1; i < (initc.size()-1); i++)
    //    cout << "params[" << i << "] = " << params[i] << "\n";
    (*arr) = make_cuDoubleComplex(0.0, 0.0);
    for(uint n = 0; n < num_modes; n++)
    {
        cout << "mode " << n << ": amp=" << initc[3*n] << " ky=" << initc[3*n+1] << ", kx=" << initc[3*n+2] << "\n";
        d_init_mode<<<1, 1>>>(arr -> get_array_d(0), layout, initc[3*n], uint(initc[3*n+1]), uint(initc[3*n+2]));
    }

//    gpuErrchk(cudaMalloc( (double**) &d_params, initc.size() * sizeof(double)));
//    gpuErrchk(cudaMemcpy(d_params, params, sizeof(double) * initc.size(), cudaMemcpyHostToDevice));

    //d_init_mode_exp<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, d_params);
    //(*arr) = make_cuDoubleComplex(0.0, 0.0);
//void d_init_mode(cuda::cmplx_t* array, cuda::slab_layout_t layout, double amp, uint col, uint row)
//    d_init_mode<<<1, 1>>>(arr -> get_array_d(0), layout, d_params);
    //cudaDeviceSynchronize();
}



// End of file initialize.cu
