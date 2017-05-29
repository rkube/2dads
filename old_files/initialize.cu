/*
 * CUDA specific initialization function
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <cmath>
#include "initialize.h"

#define EPSILON 0.000001

__global__
void d_init_lamb_dipole(cuda::real_t* array, const cuda::slab_layout_t layout, 
        const cuda::init_params_t params)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.Nx + col;

    const double x = layout.x_left + double(col) * layout.delta_x;
    const double y = layout.y_lo + double(row) * layout.delta_y;

    double r;
    double theta;
    double lbd;
    double U;
    double R;

    if((col < layout.Nx) && (row < layout.My))
    {
        r = sqrt(x * x + y * y);
        theta = atan2(x, y);
        U = params.i1;
        R = params.i2;
        // lambda = BesselZero[1,1] / R
        lbd = 3.831705970207 / R;
        if(r < R)
            array[idx] = 2 * lbd * U * j1(lbd * r) * cos(theta) / j0(3.831705970207);
        else
            array[idx] = 0.0;
    }
    return;

}


__global__ 
void d_init_sine(cuda::real_t* array, const cuda::slab_layout_t layout, const double kx, const double ky)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.Nx + col;

    const double x = layout.x_left + double(col) * layout.delta_x;
    const double y = layout.y_lo + double(row) * layout.delta_y;

    if ((col >= layout.Nx) || (row >= layout.My))
        return;
    array[idx] = sin(kx * x) + sin(ky * y);
}


__global__
void d_init_exp(cuda::real_t* array, const cuda::slab_layout_t layout, const cuda::init_params_t params)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.Nx + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.Nx) || (row >= layout.My))
        return;

    array[idx] = params.i1 + params.i2 * exp( -(x - params.i3) * (x - params.i3) / (2.0 * params.i4 * params.i4) 
                                              -(y - params.i5) * (y - params.i5) / (2.0 * params.i6 * params.i6));
}


__global__
void d_init_exp_log(cuda::real_t* array, const cuda::slab_layout_t layout, const cuda::init_params_t params)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.Nx + col;
    const double x = layout.x_left + double(col) * layout.delta_x;
    const double y = layout.y_lo + double(row) * layout.delta_y;

    if ((col >= layout.Nx) || (row >= layout.My))
        return;

    array[idx] = params.i1 + params.i2 * exp( -(x - params.i3) * (x - params.i3) / (2.0 * params.i4 * params.i4) 
                                              -(y - params.i5) * (y - params.i5) / (2.0 * params.i6 * params.i6));
    array[idx] = log(array[idx]);
}


__global__
void d_init_lapl(cuda::real_t* array, const cuda::slab_layout_t layout, const cuda::init_params_t params)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * layout.Nx + col;
    const double x = layout.x_left + double(row) * layout.delta_x;
    const double y = layout.y_lo + double(col) * layout.delta_y;

    if ((col >= layout.Nx) || (row >= layout.My))
        return;

    array[idx] = exp(- 0.5 * (x * x + y * y)/(params.i4 * params.i4)) / (params.i4 * params.i4) * 
                 ((x * x + y * y)/(params.i4 * params.i4) - 2.0);
}


/// Initialize gaussian profile around a single mode
__global__
void d_init_mode_exp(cuda::cmplx_t* array, const cuda::slab_layout_t layout, const double amp, const double modex, const double modey, const double sigma)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * (layout.Nx / 2 + 1) + col;
    double n = double(col);
    double m = double(row);
    double damp = exp ( -((n-modex)*(n-modex) / sigma) - ((m-modey)*(m-modey) / sigma) ); 
    double phase = 0.56051 * 2.0 * cuda::PI;  
    
    if ((col >= layout.Nx / 2 + 1) || (row >= layout.My))
        return;

    cuda::cmplx_t foo(damp * amp * cos(phase), damp * amp * sin(phase));
    array[idx] = foo;
    //printf("sigma = %f, re = %f, im = %f\n", sigma, array[idx].x, array[idx].y);
}


/// Initialize a single mode pointwise
__global__
void d_init_mode(cuda::cmplx_t* array, const cuda::slab_layout_t layout, const double amp, const uint row, const uint col)
{
    const uint idx = row * (layout.Nx / 2 + 1) + col;
    const double phase = 0.56051 * cuda::TWOPI;
    //CuCmplx<cuda::real_t> foo(amp * cos(phase), amp * sin(phase));
    cuda::cmplx_t foo(amp * cos(phase), amp * sin(phase));
    array[idx] = foo;
    printf("d_init_mode: mode(%d, %d) at idx = %d = (%f, %f)\n",
            row, col, idx, array[idx].re(), array[idx].im());
}


/// Initialize all modes to given value
/// Do not initialize Nyquist frequencies at col = Nx/2+1 and row = My/2+1
__global__
void d_init_all_modes(cuda::cmplx_t* array, const cuda::slab_layout_t layout, const double real, const double imag)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * (layout.Nx / 2 + 1) + col;

    if((col < layout.Nx / 2 + 1) && (row < layout.My))
    	if((col == layout.Nx / 2 ) || (row == layout.My / 2))
    		array[idx] = cuda::cmplx_t(real);
    	else
    		array[idx] = cuda::cmplx_t(real, imag);
    return;
}


// Setup PRNG http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
__global__
void d_setup_rand(curandState* state, const time_t seed, const cuda::slab_layout_t layout)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * layout.Nx + col;
	curand_init((unsigned long) seed, index, 0, &state[index]);
}


__global__
void d_init_random_modes(cuda::cmplx_t* array, curandState* state, const cuda::slab_layout_t layout)
{
	const uint row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint col = blockIdx.x * blockDim.x + threadIdx.x;
	const uint idx = row * (layout.Nx / 2 + 1) + col;

    cuda::real_t phase{0.0};
    cuda::real_t amplitude = 1000.0;// / sqrt(cuda::real_t(layout.Nx * layout.My));
    if((col < layout.Nx / 2 + 1) && (row < layout.My))
	{
		phase = curand_uniform_double(&state[idx]) * cuda::TWOPI;
		array[idx].set_re(amplitude * cos(phase));

		// Set imaginary part of nyquist frequency terms to zero
		// These frequencies are their own complex conjugate
    	if((col == layout.Nx / 2 ) || (row == layout.My / 2))
    		array[idx].set_im(0.0);
    	else
    		array[idx].set_im(amplitude * sin(phase));
	}
	return;
}

__global__
void d_zero_mode(cuda::cmplx_t* array)
{
    array[0] = cuda::cmplx_t(0.0, 0.0);
}

/// Initialize sinusoidal profile
//void init_simple_sine(cuda_array<cuda::real_t, cuda::real_t>* arr, 
void init_simple_sine(cuda_arr_real* arr,
        const vector<double> initc,
        const cuda::slab_layout_t layout)
{
    const double kx = initc[0] * cuda::TWOPI / double(layout.delta_x * double(arr -> get_nx()));
    const double ky = initc[1] * cuda::TWOPI / double(layout.delta_y * double(arr -> get_my()));

    d_init_sine<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(0), layout, kx, ky);
    cudaDeviceSynchronize();
}


/// Initialize field with a guassian profile
void init_gaussian(cuda_arr_real* arr,
        const vector<double> initc,
        const cuda::slab_layout_t layout,
        const bool log_theta)
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
void init_invlapl(cuda_arr_real* arr,
        const vector<double> initc,
        const cuda::slab_layout_t layout)
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

// Initialize all modes with random values
void init_turbulent_bath(cuda_arr_cmplx* arr, const cuda::slab_layout_t layout, const uint tlev)
{
    cuda::cmplx_t* tmp_field = new cuda::cmplx_t[layout.My * (layout.Nx / 2 + 1)];
    cuda::real_t amplitude{0.0};
    cuda::real_t phase{0.0};
    srand(time(NULL));

    // Maximum mode to initialize. Do not initialize high frequency modes
    constexpr unsigned int nmax{16};
    constexpr unsigned int mmax{16}; 

    // Loop works for My >= 32
    for(uint m = 1; m < mmax; m++)
    {
        for(uint n = 1; n < nmax; n++)
        {
            amplitude = 100. / sqrt(1. + (double(n * n + m * m) / double(nmax * mmax) * 
                                          double(n * n + m * m) / double(nmax * mmax) *
                                          double(n * n + m * m) / double(nmax * mmax) *
                                          double(n * n + m * m) / double(nmax * mmax)));
            phase = double(rand()) / double(RAND_MAX);
            //if(m == 0 || m == layout.My || n == 0 || n == layout.Nx / 2)
            //    tmp_field[m * (layout.Nx / 2 + 1) + n] = cuda::cmplx_t(0.0, 0.0);
            //else
            tmp_field[m * (layout.Nx / 2 + 1) + n] = cuda::cmplx_t(amplitude * cos(phase * 2.0 * cuda::PI), 
                                                                   amplitude * sin(phase * 2.0 * cuda::PI));
        }
    }
    tmp_field[0].set_re(0.0);
    tmp_field[0].set_im(0.0);


    gpuErrchk(cudaMemcpy(arr -> get_array_d(tlev), tmp_field, layout.My * (layout.Nx / 2 + 1) * sizeof(cuda::cmplx_t), cudaMemcpyHostToDevice));
    gpuStatus();

    ofstream of;
    of.open("tbath.dat", ios::trunc);
    of << (*arr);
    of.close();

    if(tmp_field != nullptr)
        delete [] tmp_field;
}


/// Initialize single mode from input.ini
/// Use 
/// init_function  = *_mode
/// initial_conditions = amp_0 kx_0 ky_0 sigma_0   amp_1 kx_1 ky_1 sigma_1 .... amp_N kx_N ky_N sigma_N 
/// kx -> row, ky -> column. 
/// kx is the x-mode number: kx = 0, 1, ...Nx / 2 
/// ky is the y-mode number: ky = 0... My / 2
void init_mode(cuda_arr_cmplx* arr,
        const vector<double> initc,
        const cuda::slab_layout_t layout,
        const uint tlev)
{
    const unsigned int num_modes = initc.size() / 4;
    (*arr) = CuCmplx<cuda::real_t>(0.0, 0.0);


    for(uint n = 0; n < num_modes; n++)
    {
        d_init_mode_exp<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(tlev), layout, initc[4*n], initc[4*n+1], initc[4*n+2], initc[4*n+3]);
    }

    // Zero out kx=0,ky=0 mode
    d_init_mode<<<1, 1>>>(arr -> get_array_d(tlev), layout, 0.0, 0, 0);
    gpuStatus();
}


/// Initialize a Lamb Dipole
void init_lamb(cuda_arr_real* arr,
        const vector<double> initc,
        const cuda::slab_layout_t layout)
{
    cuda::init_params_t init_params;
    init_params.i1 = (initc.size() > 1) ? initc[0] : 0.0;
    init_params.i2 = (initc.size() > 2) ? initc[1] : 0.0;
    init_params.i3 = (initc.size() > 3) ? initc[2] : 0.0;
    init_params.i4 = (initc.size() > 4) ? initc[3] : 0.0;
    init_params.i5 = (initc.size() > 5) ? initc[4] : 0.0;
    init_params.i6 = (initc.size() > 6) ? initc[5] : 0.0;

    cout << "i1 = " << init_params.i1 << ", i2 = " << init_params.i2 << ", 3 = " << init_params.i3 << endl;

    d_init_lamb_dipole<<<arr -> get_grid(), arr -> get_block()>>>(arr -> get_array_d(), layout, init_params);
    gpuStatus();
}


// End of file initialize.cu
