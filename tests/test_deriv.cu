/*
 *
 * Created on: Dec 25, 2013
 *     Author: Ralph Kube
 *
 * Helper functions for slab_cuda
 *
 */


//#include "include/common_kernel.h"
#include "include/cuda_types.h"
#include "include/cuda_array2.h"
#include "include/initialize.h"
#include <iostream>
#include <vector>
#include "cufft.h"

using namespace std;


__global__ 
void d_d_dx(cuda::cmplx_t* in, cuda::cmplx_t* out, int Nx, int My, double Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * My + col;

    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;

    const double two_pi_L = 2.0 * cuda::PI / Lx;
    if(row < Nx / 2)
        out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(row)));
    else if (row == Nx / 2)
        out[index] = make_cuDoubleComplex(0.0, 0.0);

    else if (row > Nx / 2)
        out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(row-Nx)));
}


__global__ 
void d_d_dy(cuda::cmplx_t* in, cuda::cmplx_t* out, const int Nx, const int My, const int Ly)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * My + col;

    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;

    double two_pi_l = 2.0 * cuda::PI / Ly;
    if(col < My / 2)
        out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_l * double(col)));   
    else if (col == My / 2)
        out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, 0.0));
}


__global__
void inv_laplace(cuda::cmplx_t* in, cuda::cmplx_t* out, const int Nx, const int My, const int Lx, const int Ly)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * My + col;

    // Return if this thread does not have an work item
    if ((col >= My) || (row >= Nx))
        return;
    const double four_pi_s_lx = 4.0 * cuda::PI * cuda::PI / (Lx * Lx);
    const double four_pi_s_ly = 4.0 * cuda::PI * cuda::PI / (Ly * Ly);
    cuda::cmplx_t wavenumber;

    if (row <= Nx / 2)
        wavenumber = make_cuDoubleComplex(four_pi_s_lx * cuda::real_t(col * col) + 
                                          four_pi_s_ly * cuda::real_t(row * row), 0.0);
    else if (row > Nx / 2)
        wavenumber = make_cuDoubleComplex(four_pi_s_lx * cuda::real_t((col - Nx) * (col - Nx)) +
                                          four_pi_s_ly * cuda::real_t(row * row), 0.0);

    out[index] = cuCmul(make_cuDoubleComplex(-1.0, 0.0), in[index]);
    out[index] = cuCdiv(out[index], wavenumber);
}

// Set x derivative frequencies: 0 .. Nx / 2 - 1
__global__
void set_factor_dx_lo(cuda::cmplx_t* arr, int Nx, int My, int Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * My + col;

    //const double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    arr[index] = make_cuDoubleComplex(0.0, double(row));
}

// Set x derivatives, zero frequency at Nx/2
__global__
void set_factor_dx_mid(cuda::cmplx_t* arr, int Nx, int My, int Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    //const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = Nx / 2;
    const int index = row * My + col;

    //const double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    arr[index] = make_cuDoubleComplex(-1.0, -1.0);
}


// Set x derivative frequencies Nx / 2 + 1...Nx - 1
// Row offset: Nx / 2 + 1
__global__ 
void set_factor_dx_up(cuda::cmplx_t* arr, int Nx, int My, int Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const int index = row * My + col;

    //const double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    arr[index] = make_cuDoubleComplex(0.0, double(row-Nx));

}


// Set y derivative frequencies 0...My / 2 -1
__global__
void set_factor_dy_lo(cuda::cmplx_t* arr, int Nx, int My, int Ly)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * My + col;

    if ((col >= My) || (row >= Nx))
        return;

    //arr[index] = make_cuDoubleComplex(double(blockIdx.x) double(threadIdx.x));
    arr[index] = make_cuDoubleComplex(0.0, double(col));
}



__global__
void set_factor_dy_up(cuda::cmplx_t* arr, int Nx, int My, int Ly)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= Nx) 
        return;
    
    index = (index + 1) * My - 1;
    arr[index] = make_cuDoubleComplex(0.0, 0.0);
}



// Frequencies 0 .. N/2 - 1
__global__
void d_d_dx_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, int Nx, int My, double Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * My + col;

    double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(row)));
}


// Frequencies: Nx/2
__global__
void d_d_dx_mid(cuda::cmplx_t* in, cuda::cmplx_t* out, int Nx, int My, double Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    //const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = Nx / 2;
    const int index = row * My + col;

    //const double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    out[index] = make_cuDoubleComplex(0.0, 0.0);
}


// Frequencies: Nx/2 + 1 ... Nx - 1
__global__
void d_d_dx_up(cuda::cmplx_t* in, cuda::cmplx_t* out, int Nx, int My, double Lx)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const int index = row * My + col;

    double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(row - Nx)));
}


// Frequencies 0..My / 2
__global__
void d_d_dy_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, int Nx, int My, double Ly)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * My + col;

    if ((col >= My) || (row >= Nx))
        return;
    double two_pi_L = 2.0 * cuda::PI / Ly;

    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(col)));
}


__global__
void d_d_dy_up(cuda::cmplx_t* in, cuda::cmplx_t* out, int Nx, int My, double Ly)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= Nx)
        return;

    index = (index + 1) * My - 1;
    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, 0.0));
}


void enumerate_factors_x(cuda_array<cuda::cmplx_t>& arr, cuda::slab_layout_t layout)
{
    double Lx = layout.delta_x * layout.Nx;
    // Frequencies 0...Nx/2-1
    dim3 grid_dx_half(layout.Nx / 2, arr.get_grid().y);
    // Frequencies Nx/2
    dim3 grid_dx_single(1, arr.get_grid().y);

    cout << "Enumerating\n";
    cout << "arr.grid = (" << arr.get_grid().x << ", " << arr.get_grid().y << ", " << arr.get_grid().z << ")\n";
    cout << "grid 0..N/2-1: grid = (" << grid_dx_half.x << ", " << grid_dx_half.y << ", " << grid_dx_half.z << ")\n";
    set_factor_dx_lo<<<grid_dx_half, arr.get_block()>>>(arr.get_array_d(0), layout.Nx, layout.My / 2 + 1, Lx);
    set_factor_dx_mid<<<grid_dx_single, arr.get_block()>>>(arr.get_array_d(0), layout.Nx, layout.My / 2 + 1, Lx);
    set_factor_dx_up<<<grid_dx_half, arr.get_block()>>>(arr.get_array_d(0), layout.Nx, layout.My / 2 + 1, Lx);
}

void d_dx(cuda_array<cuda::cmplx_t>& in, cuda_array<cuda::cmplx_t>& out, cuda::slab_layout_t layout)
{
    double Lx = layout.delta_x * layout.Nx;
    dim3 grid_dx_half(layout.Nx / 2, in.get_grid().y);
    dim3 grid_dx_single(1, in.get_grid().y);

    d_d_dx_lo<<<grid_dx_half, in.get_block()>>>(in.get_array_d(0), out.get_array_d(0), layout.Nx, layout.My / 2 + 1, Lx);
    d_d_dx_mid<<<grid_dx_single, in.get_block()>>>(in.get_array_d(0), out.get_array_d(0), layout.Nx, layout.My / 2 + 1, Lx);
    d_d_dx_up<<<grid_dx_half, in.get_block()>>>(in.get_array_d(0), out.get_array_d(0), layout.Nx, layout.My / 2 + 1, Lx);
}


void d_dy(cuda_array<cuda::cmplx_t>& in, cuda_array<cuda::cmplx_t>& out, cuda::slab_layout_t layout)
{
    double Ly = layout.delta_y * layout.My;
    // grid to access last column only
    dim3 block_single(cuda::cuda_blockdim_nx);
    dim3 grid_single(layout.Nx / cuda::cuda_blockdim_nx); 
    d_d_dy_lo<<<out.get_grid(), out.get_block()>>>(in.get_array_d(0), out.get_array_d(0), layout.Nx, layout.My / 2 + 1, Ly);
    d_d_dy_up<<<grid_single, block_single>>>(in.get_array_d(0), out.get_array_d(0), layout.Nx, layout.My / 2 + 1, Ly);
}


int main(void)
{
    const int Nx = 16;
    const int My = 16;
    const double xleft = -1.0;
    const double ylo = -1.0;
    const double Lx = 2.0;
    const double Ly = 2.0;
    const double delta_x = Lx / double(Nx);
    const double delta_y = Ly / double(My);
    cuda::slab_layout_t layout = {xleft, delta_x, ylo, delta_y, Nx, My};

    cufftHandle plan_fw;
    cufftHandle plan_bw;
    cufftResult err;
    size_t dft_size;

    dim3 grid = dim3(8, 1);
    dim3 block = dim3(1, 64);

    cout << "instating arr_r...\n";
    cuda_array<cuda::real_t> arr_r(1, Nx, My);
    cout << "instating arr_f...\n";
    cuda_array<cuda::cmplx_t> arr_f(1, Nx, My / 2 + 1); 

    // output arrays for derivation
    cuda_array<cuda::cmplx_t> arr_fx(1, Nx, My / 2 + 1);
    cuda_array<cuda::real_t> arr_x(1, Nx, My);

    vector<double> initc;
    initc.push_back(1.0);
    initc.push_back(0.0);

    // Fourier transform
    err = cufftPlan2d(&plan_fw, Nx, My, CUFFT_D2Z);
    cout << "Planning dft D2Z, err=" << err <<"\n";
    err = cufftPlan2d(&plan_bw, Nx, My, CUFFT_Z2D);
    cout << "Planning dft Z2D, err=" << err << "\n";
    cufftEstimate2d(Nx, My, CUFFT_D2Z, &dft_size);
    cout << "Estimated size D2Z = " << dft_size << "\n";

    //enumerate_factors_x(arr_f, layout);

    cout << "arr_f = " << arr_f;

    //////////////////////////////////////////////////////////////////////////
    // Test x -derivation
    // Initialize slab with simple sine
    //init_simple_sine(&arr_r, initc, delta_x, delta_y, xleft, ylo); 
    //cout << "arr_r = " << arr_r << "\n";

    //err = cufftExecD2Z(plan_fw, arr_r.get_array_d(0), arr_f.get_array_d(0));

    //cout << "arr_f = " << arr_f;
    //// test derivatives
    //d_dx(arr_f, arr_fx, layout);
    //cout << "arr_fx = " << arr_fx;
    //// Inverse trafo
    //err = cufftExecZ2D(plan_bw, arr_fx.get_array_d(0), arr_x.get_array_d(0));
    //// Normalize
    //arr_x *= 1. / double(Nx * My);
    //cout << "arr_x = " << arr_x;
    //////////////////////////////////////////////////////////////////////////
    // Test y-derivation
    initc.clear();
    initc.push_back(0.0);
    initc.push_back(1.0);
    init_simple_sine(&arr_r, initc, delta_x, delta_y, xleft, ylo);
    cout << "arr_r = " << arr_r;
    err = cufftExecD2Z(plan_fw, arr_r.get_array_d(0), arr_f.get_array_d(0));
    cout << "arr_f = " << arr_f;

    d_dy(arr_f, arr_fx, layout);
    cout << "arr_fx = " << arr_fx;
    err = cufftExecZ2D(plan_bw, arr_fx.get_array_d(0), arr_x.get_array_d(0));
    arr_x *= 1. / double(Nx * My);
    cout << "arr_x = " << arr_x;

    return(0);
}


// End of file slab_helper.cu
