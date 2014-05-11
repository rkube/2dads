/*
 * CUDA specific initialization function
 *
 */

#include <iostream>
#include <vector>
#include "include/initialize.h"


__global__ 
void d_init_sine(cuda::real_t* array, cuda::slab_layout_t layout, double kx, double ky)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.Nx) || (row >= layout.My))
        return;
    array[idx] = sin(kx * x) + sin(ky * y);
}


__global__
void d_init_exp(cuda::real_t* array, cuda::slab_layout_t layout, cuda::init_params_t params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.My) || (row >= layout.Nx))
        return;

    array[idx] = params.i1 + params.i2 * exp( -(x - params.i3) * (x - params.i3) / (2.0 * params.i4 * params.i4) 
                                              -(y - params.i5) * (y - params.i5) / (2.0 * params.i6 * params.i6));
}


__global__
void d_init_exp_log(cuda::real_t* array, cuda::slab_layout_t layout, cuda::init_params_t params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.My) || (row >= layout.Nx))
        return;

    array[idx] = params.i1 + params.i2 * exp( -(x - params.i3) * (x - params.i3) / (2.0 * params.i4 * params.i4) 
                                              -(y - params.i5) * (y - params.i5) / (2.0 * params.i6 * params.i6));
    array[idx] = log(array[idx]);
}


__global__
void d_init_lapl(cuda::real_t* array, cuda::slab_layout_t layout, cuda::init_params_t params)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.My + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.My) || (row >= layout.Nx))
        return;

    array[idx] = exp(- 0.5 * (x * x + y * y)/(params.i4 * params.i4)) / (params.i4 * params.i4) * 
                 ((x * x + y * y)/(params.i4 * params.i4) - 2.0);
}


/// Initialize gaussian profile around a single mode
__global__
void d_init_mode_exp(cuda::cmplx_t* array, cuda::slab_layout_t layout, double amp, double modex, double modey, double sigma)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * (layout.My / 2 + 1) + col;
    if ((col >= layout.My / 2 + 1) || (row >= layout.Nx))
        return;
    double n = double(row);
    double m = double(col);
    double damp = exp ( -((n-modex)*(n-modex) / sigma) - ((m-modey)*(m-modey) / sigma) ); 
    double phase = 0.56051 * 2.0 * cuda::PI;  
    
    cuda::cmplx_t foo(damp * amp * cos(phase), damp * amp * sin(phase));
    array[idx] = foo;
    //printf("sigma = %f, re = %f, im = %f\n", sigma, array[idx].x, array[idx].y);
}


/// Initialize a single mode pointwise
__global__
void d_init_mode(cuda::cmplx_t* array, cuda::slab_layout_t layout, double amp, uint row, uint col)
{
    const uint idx = row * (layout.My / 2 + 1) + col;
    const double phase = 0.56051 * cuda::TWOPI;
    CuCmplx<cuda::real_t> foo(amp * cos(phase), amp * sin(phase));
    array[idx] = foo;
    printf("d_init_mode: mode(%d, %d) at idx = %d = (%f, %f)\n",
            row, col, idx, cos(phase), sin(phase));
}


/// Initialize all modes to given value
__global__
void d_init_all_modes(cuda::cmplx_t* array, cuda::slab_layout_t layout, double real, double imag)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * (layout.My / 2 + 1) + col;
    if ((col >= layout.My / 2 + 1) || (row >= layout.Nx))
        return;

    array[idx] = cuda::cmplx_t(real, imag);
}


/// Initialize sinusoidal profile
//void init_simple_sine(cuda_array<cuda::real_t, cuda::real_t>* arr, 
void init_simple_sine(cuda_arr_real* arr,
        vector<double> initc,
        cuda::slab_layout_t layout)
{
    dim3 grid = arr -> get_grid();
    dim3 block = arr -> get_block();

    const double kx = initc[0] * cuda::TWOPI / double(layout.delta_x * double(arr -> get_nx()));
    const double ky = initc[1] * cuda::TWOPI / double(layout.delta_y * double(arr -> get_my()));

    d_init_sine<<<grid, block>>>(arr -> get_array_d(0), layout, kx, ky);
    cudaDeviceSynchronize();
}


/// Initialize field with a guassian profile
void init_gaussian(cuda_array<cuda::real_t, cuda::real_t>* arr,
        vector<double> initc,
        cuda::slab_layout_t layout,
        bool log_theta)
{
    cuda::init_params_t init_params;

    init_params.i1 = (initc.size() > 0) ? initc[0] : 0.0;
    init_params.i2 = (initc.size() > 1) ? initc[1] : 0.0;
    init_params.i3 = (initc.size() > 2) ? initc[2] : 0.0;
    init_params.i4 = (initc.size() > 3) ? initc[3] : 0.0;
    init_params.i5 = (initc.size() > 4) ? initc[4] : 0.0;
    init_params.i6 = (initc.size() > 5) ? initc[5] : 0.0;

    if (log_theta)
        d_init_exp_log<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, init_params);
    else
        d_init_exp<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, init_params);
    cudaDeviceSynchronize();
}


/// Initialize real field with nabla^2 exp(-(x-x0)^2/ (2. * sigma^2) - (y - y0)^2 / (2. * sigma_y^2)
void init_invlapl(cuda_array<cuda::real_t, cuda::real_t>* arr,
        vector<double> initc,
        cuda::slab_layout_t layout)
{

    cuda::init_params_t init_params;
    init_params.i1 = (initc.size() > 1) ? initc[0] : 0.0;
    init_params.i2 = (initc.size() > 2) ? initc[1] : 0.0;
    init_params.i3 = (initc.size() > 3) ? initc[2] : 0.0;
    init_params.i4 = (initc.size() > 4) ? initc[3] : 0.0;
    init_params.i5 = (initc.size() > 5) ? initc[4] : 0.0;
    init_params.i6 = (initc.size() > 6) ? initc[5] : 0.0;

    d_init_lapl<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, init_params);
    cudaDeviceSynchronize();
}

// Initialize all modes with constant value
void init_all_modes(cuda_arr_cmplx* arr, vector<double> initc, cuda::slab_layout_t layout, uint tlev)
{
    double real = initc[0];
    double imag = initc[1];
d_init_all_modes<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(tlev), layout, real, imag);
}


/// Initialize single mode from input.ini
/// Use 
/// init_function  = *_mode
/// initial_conditions = amp_0 kx_0 ky_0 sigma_0   amp_1 kx_1 ky_1 sigma_1 .... amp_N kx_N ky_N sigma_N 
/// kx -> row, ky -> column. 
/// kx is the x-mode number: kx = 0, 1, ...Nx -1
/// ky is the y-mode number: ky = 0... My / 2
void init_mode(cuda_array<cuda::cmplx_t, cuda::real_t>* arr,
        vector<double> initc,
        cuda::slab_layout_t layout,
        uint tlev)
{
    const unsigned int num_modes = initc.size() / 4;
    (*arr) = CuCmplx<cuda::real_t>(0.0, 0.0);

#ifdef DEBUG
    cout << "Initializing modes:\n";
#endif
    for(uint n = 0; n < num_modes; n++)
    {
#ifdef DEBUG
        cout << "mode " << n << ": amp=" << initc[4*n] << " kx=" << initc[4*n+1] << ", ky=" << initc[4*n+2] << ", sigma=" << initc[4*n+3] << "\n";
#endif
        d_init_mode_exp<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(tlev), layout, initc[4*n], initc[4*n+1], initc[4*n+2], initc[4*n+3]);
        //d_init_mode<<<1, 1>>>(arr -> get_array_d(0), layout, initc[4*n], uint(initc[4*n+1]), uint(initc[4*n+2])); 
    }
}

// End of file initialize.cu
