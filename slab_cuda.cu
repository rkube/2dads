//#include <algorithm>
#include "error.h"
#include "slab_cuda.h"

using namespace std;
/*****************************************************************************
 *
 * Kernel implementation
 *
 ****************************************************************************/

//    d/dy: Compute spectral y-derivative for frequencies 0 .. My/2 - 1,
//          stored in the columns 0..My/2-1
//    Call:  d_d_dy_lo<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
//    block_nx21    = dim3(cuda::blockdim_x, 1)
//    grid_dy_half  = dim3(((Nx / 2 + 1) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, My / 2)
__global__
void d_d_dy_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    double two_pi_L = cuda::TWOPI / Ly;
    // Return if we don't have an item to work on
    if((col < Nx21) && (row < My / 2))
    	out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * double(row));

    return;
    //if((col >= Nx21) || (row >= My / 2))
    //    return;
}


__global__
void d_d_dy_lo_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if((col < Nx21) && (row < My / 2))
    	arr[index] = cuda::cmplx_t(1000 + row, col);
    return;
    //if((col >= Nx21) || (row >= My / 2))
    //    return;
}


// Frequencies: My/2, stored in row My / 2
__global__
void d_d_dy_mid(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if((col < Nx21) && (row == My / 2))
        out[index] = cuda::cmplx_t(0.0, 0.0);
   	return;
    // Return if we don't have an item to work on
    //if(col >= Nx21)
    //    return;
    //
}


__global__
void d_d_dy_mid_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if((col < Nx21) && (row == My / 2))
    	arr[index] = cuda::cmplx_t(2000 + row, col);
   	return;
    //if(col >= Nx21)
    //    return;
//
    //arr[index] = cuda::cmplx_t(2000 + row, col);
}


// Frequencies: My/2 + 1 ... My - 1
// These are stored in the last My/2-1 rows
__global__
void d_d_dy_up(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    double two_pi_L = cuda::TWOPI / Ly;
    if((col < Nx21) && (row < My))
        out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * (double(row) - double(My)));
    return;

    // Return if we don't have an item to work on
    //if((col >= Nx21) || (row >= My))
    //    return;
    //
    //out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * (double(row) - double(My)));
}


__global__
void d_d_dy_up_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if((col < Nx21) && (row < My))
        arr[index] = cuda::cmplx_t(3000.0 + double(row) - double(My), col);
    return;

    //if((col >= Nx21) || (row >= My))
    //    return;

    //arr[index] = cuda::cmplx_t(3000.0 + double(row) - double(My), col);
}


// x derivation
// Frequencies 0..Nx / 2, stored in cols 0..Nx/2-1
__global__
void d_d_dx_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Lx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;
    double two_pi_L = cuda::TWOPI / Lx;

    if((row < My) && (col < Nx21 - 1))
        out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * double(col));
    //if ((col >= Nx21 - 1) || (row >= My))
    //    return;
    //(a + ib) * ik = -(b * k) + i(a * k)
    //out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * double(col));
}


__global__
void d_d_dx_lo_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if((row < My) && (col < Nx21 - 1))
        arr[index] = cuda::cmplx_t(double(row), double(col));

    return;
    //if ((col >= Nx21 - 1) || (row >= My))
    //    return;
    //
    //arr[index] = cuda::cmplx_t(double(row), double(col));
}



// x derivation
// Frequencies Nx/2, stored in the last column. Set them to zero
__global__
void d_d_dx_up(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Lx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x + Nx21 - 1;
    const uint index = row * Nx21 + col;

    if ((row < My) && (col == Nx21 - 1))
    	out[index] = cuda::cmplx_t(0.0, 0.0);
}


__global__
void d_d_dx_up_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x + Nx21 - 1;
    const uint index = row * Nx21 + col;

    if((row < My) && (col == Nx21 - 1))
    	arr[index] = cuda::cmplx_t(row, double(1000 + col));
    return;
}



// ky=0 modes are stored in the first row, Nx21 columns
__global__
void d_kill_ky0(cuda::cmplx_t* in, const uint My, const uint Nx21)
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

//    if (col >= Nx21)
//        return;
//
    if(col < Nx21)    
        in[col] = cuda::cmplx_t(0.0, 0.0);
}


// invert two dimensional laplace equation.
// In spectral space, 
//                              / 4 pi^2 ((ky/Ly)^2 + (kx/Lx)^2 )  for ky, kx  <= N/2
// phi(ky, kx) = omega(ky, kx)  
//                              \ 4 pi^2 (((ky-My)/Ly)^2 + (kx/Lx)^2) for ky > N/2 and kx <= N/2
//
// and phi(0,0) = 0 (to avoid division by zero)
// Divide into 4 sectors:
//
//            Nx/2    1 (last element)
//         ^<------>|------|
//  My/2+1 |        |      |
//         |   I    | III  |
//         |        |      |
//         v        |      |
//         =================
//         ^        |      |
//         |        |      |
//  My/2-1 |  II    |  IV  |   
//         |        |      |
//         v<------>|------|
//           Nx/2      1
//
// 
// sector I    : ky <= My/2, kx < Nx/2   BS = (cuda_blockdim_nx, 1), GS = (((Nx / 2 + 1) + cuda_blockdim_nx - 1) / cuda_blockdim_nx, My / 2 + 1)
// sector II   : ky >  My/2, kx < Nx/2   BS = (cuda_blockdim_nx, 1), GS = (((Nx / 2 - 1) + cuda_blockdim_nx - 1) / cuda_blockdim_nx, My / 2 - 1)
// sector III  : ky <= My/2, kx = Nx/2   BS = (cuda_blockdim_nx, 1), GS = (1, My / 2 + 1) (This wastes cuda_blockdim_nx - 1 threads in each thread)
// sector IV   : ky >  My/2, kx = Nx/2   BS = (cuda_blockdim_nx, 1), GS = (1, My / 2 - 1) (This wastes cuda_blockdim_nx - 1 threads in each thread)
//
// Pro: wavenumbers can be computed from index without if-else blocks
// Con: Diverging memory access

__global__
void d_inv_laplace_sec1(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((col >= Nx21) || (row >= My / 2))
        return;

    out[idx] = in[idx] / cuda::cmplx_t(-cuda::FOURPIS * (double(col * col) * inv_Lx2 + double(row * row) * inv_Ly2), 0.0);
}


__global__
void d_inv_laplace_sec1_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((col >= Nx21) || (row >= My / 2))
        return;
    
    arr[idx] = cuda::cmplx_t(1000 + row, col);
}


__global__
void d_inv_laplace_sec2(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((col >= Nx21) || (row >= My))
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            ((double(row) - double(My)) * (double(row) - double(My))) * inv_Ly2 +
            (double(col * col) * inv_Lx2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec2_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((col >= Nx21) || (row > My - 1))
        return;

    arr[idx] = cuda::cmplx_t(2000 + row - My, col);
}



// Pass Nx = Nx and My = My / 2 +1 for correct indexing
__global__
void d_inv_laplace_sec3(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 
    if (row >= My / 2)
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            (double(row * row) * inv_Ly2 + double(col * col) * inv_Lx2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec3_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 
    if (row >= My / 2)
        return;

    arr[idx] = cuda::cmplx_t(3000 + row, col);
}


// Pass Nx = Nx and My = My / 2 + 1 for correct indexing
__global__
void d_inv_laplace_sec4(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 

    if (row >= My)
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            ((double(row) - double(My)) * (double(row) - double(My)) * inv_Ly2 + double(col * col) * inv_Lx2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec4_enumerate(cuda::cmplx_t* arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 

    if (row >= My)
        return;

    arr[idx] = cuda::cmplx_t(4000 + row - My, col);
}


__global__
void d_inv_laplace_zero(cuda::cmplx_t* out)
{
    out[0] = cuda::cmplx_t(0.0, 0.0);
}


/*
 * Stiffly stable time integration
 * temp = sum(k=1..level) alpha[T-2][level-k] * u^{T-k} + delta_t * beta[T-2][level -k - 1] * u_RHS^{T-k-1}
 * u^{n+1}_{i} = temp / (alpha[T-2][0] + delta_t * diff * (kx^2 + ky^2))
 *
 * Use same sector splitting as for inv_laplace
 */


__global__
void d_integrate_stiff_sec1_enumerate(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev) 

{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.Nx + col;
    if ((row >= p.My) || (col >= p.Nx))
        return;

    A_rhs[p.level - tlev][idx] = cuda::cmplx_t(1000 + row, col);
}


__global__
void d_integrate_stiff_sec1(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev) 

{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.Nx + col;
    if ((col >= p.Nx) || (row >= p.My))
        return;

    unsigned int off_a = (tlev - 2) * p.level + tlev;
    unsigned int off_b = (tlev - 2) * (p.level - 1) + tlev- 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(row) * cuda::TWOPI / p.length_y;
    cuda::real_t kx2 = kx * kx + ky * ky;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * kx2 + p.hv * kx2 * kx2 * kx2));
    

    // Add contribution from explicit and implicit parts
    for(uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div; 
}


__global__
void d_integrate_stiff_sec2_enumerate(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + p.My / 2 +  1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.Nx + col;
    if ((col >= p.Nx) || (row >= p.My))
        return;

    A_rhs[p.level - tlev][idx] = cuda::cmplx_t(2000 + row - p.My, col);
}

__global__
void d_integrate_stiff_sec2(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + p.My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.Nx + col;
    if ((col >= p.Nx) || (row >= p.My))
        return;

    unsigned int off_a = (tlev - 2) * p.level + tlev;
    unsigned int off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = (cuda::real_t(row) - cuda::real_t(p.My)) * cuda::TWOPI / p.length_y;
    cuda::real_t k2 = kx * kx + ky * ky;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

    for(uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) *temp_div;
}


__global__
void d_integrate_stiff_sec3_enumerate(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = p.Nx - 1;
    const uint idx = row * p.Nx + col; 
    if(row > p.My / 2 + 1)
        return;

    A_rhs[p.level - tlev][idx] = cuda::cmplx_t(3000 + row, col);
}

__global__
void d_integrate_stiff_sec3(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = p.Nx - 1;
    const uint idx = row * p.Nx + col; 
    if(row > p.My / 2 + 1)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(row) * cuda::TWOPI/ p.length_y;
    cuda::real_t k2 = kx * kx + ky * ky;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));
    for(uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
}


__global__
void d_integrate_stiff_sec4_enumerate(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + p.My / 2 + 1;
    const uint col = p.Nx - 1;
    const uint idx = row * p.Nx + col; 
    if (col >= p.Nx)
        return;

    cuda::cmplx_t value(4000 + row - p.My, col);
    A_rhs[p.level - tlev][idx] = value;
}

__global__
void d_integrate_stiff_sec4(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + p.My / 2 + 1;
    const uint col = p.Nx - 1;
    const uint idx = row * p.Nx + col; 
    if (row >= p.Nx)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = (cuda::real_t(row) - cuda::real_t(p.My)) * cuda::TWOPI / p.length_y;
    cuda::real_t k2 = kx * kx + ky * ky;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

    for(uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
}


__global__
void d_integrate_stiff_ky0(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = col;
    if (col >= p.Nx)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    // ky = 0
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    // ky = 0
    cuda::real_t temp_div =  1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * kx * kx + p.hv * kx * kx * kx * kx * kx * kx));

    for (uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;

}
// Print very verbose debug information of what stiffk does
// Do no update A!
__global__
void d_integrate_stiff_debug(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev, uint row, uint col)
{
    col = col % p.Nx;
    row = row % p.My;


    const uint idx = row * p.Nx + col;
    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::TWOPI * cuda::real_t(col) / p.length_x;
    cuda::real_t ky = cuda::TWOPI * cuda::real_t(row) / p.length_y;
    cuda::real_t k2 = kx * kx + ky * ky;

    printf("----d_integrate_stiff_debug:\n");
    printf("p.Nx = %d, p.My = %d\n", p.Nx, p.My);
    printf("p.level = %d, tlev = %d, kx = %5.3f, ky = %5.3f\n", p.level, tlev, kx, ky);
    printf("row = %d, col = %d\n", row, col);
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev + 1, idx, A[p.level - tlev + 1][idx].re(), A[p.level - tlev + 1][idx].im());
    printf("delta_t = %f, diff = %f hv = %f, kx = %f, ky = %f\n", p.delta_t, p.diff, p.hv, kx, ky);
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));
    cuda::cmplx_t result(0.0, 0.0);

    printf("sum_alpha = (%f, %f)\n", sum_alpha.re(), sum_alpha.im());
    printf("sum_beta = (%f, %f)\n", sum_beta.re(), sum_beta.im());
    printf("\ttlev = %d, off_a = %d, off_b = %d\n", tlev, off_a, off_b);
    for(uint k = 1; k < tlev; k++)
    {
        printf("\ttlev=%d, k = %d\t %f * A[%d] + dt * %f * A_R[%d]\n", tlev, k, alpha[off_a - k], p.level - k, beta[off_b - k], p.level - 1 - k);
        printf("\ttlev=%d, k = %d\t sum_alpha = (%f, %f) + %f * (%f, %f)\n", tlev, k, sum_alpha.re(), sum_alpha.im(), alpha[off_a - k], (A[p.level -k][idx]).re(), (A[p.level -k][idx]).im());
        printf("\ttlev=%d, k = %d\t sum_beta = (%f, %f) + %f * (%f, %f)\n", tlev, k, sum_beta.re(), sum_beta.im(), beta[off_b - k], (A_rhs[p.level - 1 - k][idx]).re(), (A_rhs[p.level - 1 - k][idx]).im());
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    result = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
    printf("\ttlev=%d, computing A[%d], gamma_0 = %f\n", tlev, p.level - tlev, alpha[(tlev - 2) * p.level]);
    printf("sum1_alpha = (%f, %f)\tsum1_beta = (%f, %f)\t", sum_alpha.re(), sum_alpha.im(), sum_beta.re(),  sum_beta.im());
    printf("temp_div = %f\n", temp_div); 
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev, idx, result.re(), result.im());
    printf("\n");
}


/*
 *
 * Kernels to compute non-linear operators
 *
 */


/// @brief Poisson brackt: {f, phi} = d_dx(f d_dy(g)) - d_dy(f d_dx(g)) = f_x g_y - f_y g_x
/// @param theta_x: f_x
/// @param theta_y: f_y
/// @param strmf_x: phi_x
/// @param strmf_y: phi_y
/// @param out: Field to store result in
/// @param Nx: Number of modes in x-direction (obs! field has Nx/2-1 columns)
/// @param My: Number of modes in y-direction
/// @detailed Poisson bracket is defined as d_y(f * phi_x) - d_x(f * phi_y)
/// @detailed When used to describe ExB advection the time derivative operator is partial f / partial t + {phi, f} = ...
/// @detailed In numerics, the Poisson bracket goes into the non-linear part of the time evolution equation:
/// @detailed df/dt + {phi, f} = ... => df/dt = {f, phi} + ...
/// @detailed
__global__
void d_pbracket(cuda::real_t* f_x, cuda::real_t* f_y, cuda::real_t* g_x, cuda::real_t* g_y, cuda::real_t* out, const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    if ((row >= My) || (col >= Nx))
       return;
    out[idx] = f_x[idx] * g_y[idx] - f_y[idx] * g_x[idx];
}


// RHS for logarithmic density field:
// theta_x * strmf_x - theta_y * strmf_x + diff * (theta_x^2 + theta_y^2)
__global__
void d_theta_rhs_log(cuda::real_t* theta_x, cuda::real_t* theta_y, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t diff, cuda::real_t* tmp_arr, const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;
    if ((row >= My) || (col >= Nx))
       return;

    tmp_arr[idx] = theta_x[idx] * strmf_y[idx] - theta_y[idx] * strmf_x[idx] + diff * (theta_x[idx] * theta_x[idx] + theta_y[idx] * theta_y[idx]);
}


__global__
void d_theta_rhs_hw(cuda::cmplx_t* theta_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::cmplx_t* strmf_y_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((row >= My) || (col >= Nx21))
        return;
    
    theta_rhs_hat[idx] += ((strmf_hat[idx] - theta_hat[idx]) * C) - strmf_y_hat[idx];
}


__global__
void d_theta_rhs_hw_debug(cuda::cmplx_t* theta_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::cmplx_t* strmf_y_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint row = 0;
    const uint col = 0;
    const uint idx = row * Nx21 + col;

    printf("d_theta_rhs_hw_debug: initially: theta_rhs_hat[%d] = (%f, %f)\n", idx, theta_rhs_hat[idx].re(), theta_rhs_hat[idx].im());
    cuda::cmplx_t dummy = (theta_rhs_hat[idx] + (strmf_hat[idx] - theta_hat[idx]) * C - strmf_y_hat[idx]);
    printf("                             --> theta_rhs_hat[%d] = (%f, %f)\tC = %f, strmf_hat = (%f, %f), theta_hat =(%f, %f), strmf_y_hat=(%f,%f)\n" ,
            idx, dummy.re(), dummy.im(), C, strmf_hat[idx].re(), strmf_hat[idx].im(), theta_hat[idx].re(), theta_hat[idx].im(), 
            strmf_y_hat[idx].re(), strmf_y_hat[idx].im()); 
}


// RHS for vorticity eq, interchange turbulence
// RHS = RHS - int * theta_y - sdiss * strmf - collfric * omega
__global__
void d_omega_ic(cuda::cmplx_t* theta_y_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* omega_hat, const cuda::real_t ic, const cuda::real_t sdiss, const cuda::real_t cfric, cuda::cmplx_t* out, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((row >= My) || (col >= Nx21))
       return;

    out[idx] = out[idx] - theta_y_hat[idx] * ic + strmf_hat[idx] * sdiss + omega_hat[idx] * cfric;
}


__global__
void d_omega_rhs_hw_debug(cuda::cmplx_t* omega_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint row = 0;
    const uint col = 0;
    const uint idx = row * Nx21 + col;

    printf("d_omega_rhs_hw_debug: initially: omega_rhs_hat[%d] = (%f, %f)\n", idx, omega_rhs_hat[idx].re(), omega_rhs_hat[idx].im());
    cuda::cmplx_t dummy = omega_rhs_hat[idx] + (strmf_hat[idx] - theta_hat[idx]) * C;
    printf("                             --> omega_rhs_hat[%d] = (%f, %f)\tC = %f, strmf_hat = (%f, %f), theta_hat = (%f, %f)\n",
            idx, dummy.re(), dummy.im(), C, strmf_hat[idx].re(), strmf_hat[idx].im(), theta_hat[idx].re(), theta_hat[idx].im());
}

__global__
void d_omega_rhs_hw(cuda::cmplx_t* omega_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((row >= My) || (col >= Nx21))
       return;

    omega_rhs_hat[idx] += (strmf_hat[idx] - theta_hat[idx]) * C;
}


__global__
void d_coupling_hwmod(cuda::cmplx_t* rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    // Start row with offset 1, this skips all ky=0 modes
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((row >= My) || (col >= Nx21))
        return;

    cuda::cmplx_t dummy = (strmf_hat[idx] - theta_hat[idx]) * C;
    rhs_hat[idx] += dummy;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

map<twodads::rhs_t, slab_cuda::rhs_fun_ptr> slab_cuda :: rhs_func_map = slab_cuda::create_rhs_func_map();

/// Initializes all real and fourier fields
/// Sets pointer to RHS functions for theta and omega
slab_cuda :: slab_cuda(slab_config my_config) :
    config(my_config),
    My(my_config.get_my()),
    Nx(my_config.get_nx()),
    tlevs(my_config.get_tlevs()),
    theta(1, My, Nx), theta_x(1, My, Nx), theta_y(1, My, Nx),
    omega(1, My, Nx), omega_x(1, My, Nx), omega_y(1, My, Nx),
    strmf(1, My, Nx), strmf_x(1, My, Nx), strmf_y(1, My, Nx),
    tmp_array(1, My, Nx), 
    theta_rhs(1, My, Nx), omega_rhs(1, My, Nx),
    theta_hat(tlevs, My, Nx / 2 + 1), theta_x_hat(1, My, Nx / 2 + 1), theta_y_hat(1, My, Nx / 2 + 1),
    omega_hat(tlevs, My, Nx / 2 + 1), omega_x_hat(1, My, Nx / 2 + 1), omega_y_hat(1, My, Nx / 2 + 1),
    strmf_hat(1, My, Nx / 2 + 1), strmf_x_hat(1, My, Nx / 2 + 1), strmf_y_hat(1, My, Nx / 2 + 1),
    tmp_array_hat(1, My, Nx / 2 + 1), 
    theta_rhs_hat(tlevs - 1, My, Nx / 2 + 1),
    omega_rhs_hat(tlevs - 1, My, Nx / 2 + 1),
    dft_is_initialized(init_dft()),
    stiff_params(config.get_deltat(), config.get_lengthx(), config.get_lengthy(), config.get_model_params(0),
            config.get_model_params(1), My, Nx / 2 + 1, tlevs),
    slab_layout(config.get_xleft(), config.get_deltax(), config.get_ylow(), config.get_deltay(), My, Nx),
    block_my_nx(theta.get_block()),
    grid_my_nx(theta.get_grid()),
    block_my_nx21(theta_hat.get_block()),
    grid_my_nx21(theta_hat.get_grid()),
    block_nx21(dim3(cuda::blockdim_nx, 1)),
    grid_nx21_sec1(dim3( ((Nx / 2 + 1) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, My / 2 + 1)),
    grid_nx21_sec2(dim3( ((Nx / 2 + 1) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, My / 2 - 1)),
    grid_nx21_sec3(dim3(1, My / 2 + 1)),
    grid_nx21_sec4(dim3(1, My / 2 - 1)),
    grid_dx_half(dim3(((Nx / 2) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, theta_hat.get_grid().y)),
    grid_dx_single(dim3(1, theta_hat.get_grid().y)),
    grid_dy_half(dim3(((Nx / 2) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, My / 2)),
    grid_dy_single(dim3(((Nx / 2) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, 1))
{
    theta_rhs_func = rhs_func_map[config.get_theta_rhs_type()];
    omega_rhs_func = rhs_func_map[config.get_omega_rhs_type()];

    rhs_array_map[twodads::f_theta_hat] = &theta_rhs_hat;
    rhs_array_map[twodads::f_omega_hat] = &omega_rhs_hat;

    get_field_by_name[twodads::f_theta] = &theta;
    get_field_by_name[twodads::f_theta_x] = &theta_x;
    get_field_by_name[twodads::f_theta_y] = &theta_y;
    get_field_by_name[twodads::f_omega] = &omega;
    get_field_by_name[twodads::f_omega_x] = &omega_x;
    get_field_by_name[twodads::f_omega_y] = &omega_y;
    get_field_by_name[twodads::f_strmf] = &strmf;
    get_field_by_name[twodads::f_strmf_x] = &strmf_x;
    get_field_by_name[twodads::f_strmf_y] = &strmf_y;
    get_field_by_name[twodads::f_tmp] = &tmp_array;
    get_field_by_name[twodads::f_theta_rhs] = &theta_rhs;
    get_field_by_name[twodads::f_omega_rhs] = &omega_rhs;

    get_field_k_by_name[twodads::f_theta_hat] = &theta_hat;
    get_field_k_by_name[twodads::f_theta_x_hat] = &theta_x_hat;
    get_field_k_by_name[twodads::f_theta_y_hat] = &theta_y_hat;
    get_field_k_by_name[twodads::f_omega_hat] = &omega_hat;
    get_field_k_by_name[twodads::f_omega_x_hat] = &omega_x_hat;
    get_field_k_by_name[twodads::f_omega_y_hat] = &omega_y_hat;
    get_field_k_by_name[twodads::f_strmf_hat] = &strmf_hat;
    get_field_k_by_name[twodads::f_strmf_x_hat] = &strmf_x_hat;
    get_field_k_by_name[twodads::f_strmf_y_hat] = &strmf_y_hat;
    get_field_k_by_name[twodads::f_theta_rhs_hat] = &theta_rhs_hat;
    get_field_k_by_name[twodads::f_omega_rhs_hat] = &omega_rhs_hat;
    get_field_k_by_name[twodads::f_tmp_hat] = &tmp_array_hat;

    get_output_by_name[twodads::o_theta] = &theta;
    get_output_by_name[twodads::o_theta_x] = &theta_x;
    get_output_by_name[twodads::o_theta_y] = &theta_y;
    get_output_by_name[twodads::o_omega] = &omega;
    get_output_by_name[twodads::o_omega_x] = &omega_x;
    get_output_by_name[twodads::o_omega_y] = &omega_y;
    get_output_by_name[twodads::o_strmf] = &strmf;
    get_output_by_name[twodads::o_strmf_x] = &strmf_x;
    get_output_by_name[twodads::o_strmf_y] = &strmf_y;
    get_output_by_name[twodads::o_theta_rhs] = &theta_rhs;
    get_output_by_name[twodads::o_omega_rhs] = &omega_rhs;

    //* Copy coefficients alpha and beta for time integration scheme to device */
    gpuErrchk(cudaMalloc((void**) &d_ss3_alpha, sizeof(cuda::ss3_alpha_r)));
    gpuErrchk(cudaMemcpy(d_ss3_alpha, &cuda::ss3_alpha_r[0], sizeof(cuda::ss3_alpha_r), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**) &d_ss3_beta, sizeof(cuda::ss3_beta_r)));
    gpuErrchk(cudaMemcpy(d_ss3_beta, &cuda::ss3_beta_r[0], sizeof(cuda::ss3_beta_r), cudaMemcpyHostToDevice));
}


slab_cuda :: ~slab_cuda()
{
    gpuErrchk(cudaFree(d_ss3_alpha));
    gpuErrchk(cudaFree(d_ss3_beta));
    finish_dft();
}


bool slab_cuda :: init_dft()
{
    cufftResult err;
    err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z);
    if (err != 0)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << "\n";
        throw gpu_error(err_str.str());
    }

    err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D);
    if (err != 0)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << "\n";
        throw gpu_error(err_str.str());
    }
    return(true);
}


// Cleanup cuFFT
void slab_cuda :: finish_dft()
{
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
}

// check config consistency
void slab_cuda :: test_slab_config()
{
    config.consistency();
}

/// @brief initialize slab to start with time integration 
void slab_cuda :: initialize()
{
    switch(config.get_init_function())
    {
        // In this switch block we compute all initial fields (theta, omega and/or strmf)
        // After this block the slab is in the following state:
        // * theta, omega and/or strmf are in the desired initial state
        // * theta_hat, omega_hat are initialized for tlev - 1
        // * strmf_hat is initialized at tlev=0 (since it is not a dynamic field)
        // * spatial derivatives have not been computed 
        // * RHS has not been evaluated
        case twodads::init_NA:
            // This should never run. init_NA throws an error in slab.consistency() :)
            break;

        case twodads::init_theta_gaussian:
            init_gaussian(&theta, config.get_initc(), slab_layout, config.get_log_theta());
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_both_gaussian:
            init_gaussian(&omega, config.get_initc(), slab_layout, 0);
            dft_r2c(twodads::f_omega, twodads::f_omega_hat, config.get_tlevs() - 1);

            init_gaussian(&theta, config.get_initc(), slab_layout, config.get_log_theta());
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_theta_mode:
            init_mode(&theta_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);
            dft_c2r(twodads::f_theta_hat, twodads::f_theta, config.get_tlevs() - 1);

            break;

        case twodads::init_omega_mode:
            init_mode(&omega_hat, config.get_initc(), slab_layout, config.get_tlevs()  - 1);

            // Compute stream function and spatial derivatives for omega, and phi
            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_both_mode:
            init_mode(&theta_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);
            init_mode(&omega_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);

            break;

        case twodads::init_theta_sine:
            init_simple_sine(&theta, config.get_initc(), slab_layout);
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_omega_sine:
            init_simple_sine(&omega, config.get_initc(), slab_layout);
            dft_r2c(twodads::f_omega, twodads::f_omega_hat, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_both_sine:
            init_simple_sine(&theta, config.get_initc(), slab_layout);
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);
            init_simple_sine(&omega, config.get_initc(), slab_layout);
            dft_r2c(twodads::f_omega, twodads::f_omega_hat, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_test:
            //init_invlapl(&theta, config.get_initc(), slab_layout);
            //dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);
            //move_t(twodads::f_theta_hat, config.get_tlevs() - 1, 0);
            init_all_modes(&omega_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);
            dft_c2r(twodads::f_omega_hat, twodads::f_omega, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_file:
            break;
    }

    // Compute spatial derivatives and RHS
    d_dx(twodads::f_theta_hat, twodads::f_theta_x_hat, config.get_tlevs() - 1);
    d_dx(twodads::f_omega_hat, twodads::f_omega_x_hat, config.get_tlevs() - 1);
    d_dx(twodads::f_strmf_hat, twodads::f_strmf_x_hat, 0);
    d_dy(twodads::f_theta_hat, twodads::f_theta_y_hat, config.get_tlevs() - 1);
    d_dy(twodads::f_omega_hat, twodads::f_omega_y_hat, config.get_tlevs() - 1);
    d_dy(twodads::f_strmf_hat, twodads::f_strmf_y_hat, 0);

    dft_c2r(twodads::f_theta_hat, twodads::f_theta, config.get_tlevs() - 1);
    dft_c2r(twodads::f_theta_x_hat, twodads::f_theta_x, 0);
    dft_c2r(twodads::f_theta_y_hat, twodads::f_theta_y, 0);

    //cout << "slab_cuda::initialize" << endl;
    //cout << "theta_hat = " << theta_hat << endl;
    //cout << "theta = " << theta << endl;
    //print_field(twodads::f_theta, "theta2.dat");

    //cout << "theta_x_hat = " << theta_x_hat << endl;
    //cout << "theta_x = " << theta_x << endl;
    //print_field(twodads::f_theta_x, "theta_x.dat");

    //cout << "theta_y_hat = " << theta_y_hat << endl;
    //cout << "theta_y = " << theta_y << endl;
    //print_field(twodads::f_theta_y, "theta_y.dat");


    dft_c2r(twodads::f_omega_hat, twodads::f_omega, config.get_tlevs() - 1);
    dft_c2r(twodads::f_omega_x_hat, twodads::f_omega_x, 0);
    dft_c2r(twodads::f_omega_y_hat, twodads::f_omega_y, 0);

    dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
    dft_c2r(twodads::f_strmf_x_hat, twodads::f_strmf_x, 0);
    dft_c2r(twodads::f_strmf_y_hat, twodads::f_strmf_y, 0);

    // Compute RHS from last time levels
    rhs_fun(config.get_tlevs() - 1);
    move_t(twodads::f_theta_rhs_hat, config.get_tlevs() - 2, 0);
    move_t(twodads::f_omega_rhs_hat, config.get_tlevs() - 2, 0);
}


// Move data from time level t_src to t_dst
void slab_cuda :: move_t(twodads::field_k_t fname, uint t_dst, uint t_src)
{
    //cuda_array<cuda::cmplx_t>* arr = get_field_by_name(fname);
    cuda_arr_cmplx* arr = get_field_k_by_name[fname];
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> move(t_dst, t_src);
}


// Copy data from time level t_src to t_dst
void slab_cuda :: copy_t(twodads::field_k_t fname, uint t_dst, uint t_src)
{
    //cuda_array<cuda::cmplx_t>* arr = get_field_by_name(fname);
    cuda_arr_cmplx* arr = get_field_k_by_name[fname];
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> copy(t_dst, t_src);
}


// Set fname to a constant value at time index tlev
void slab_cuda::set_t(twodads::field_k_t fname, cuda::cmplx_t val, uint t_src)
{
    //cuda_array<cuda::cmplx_t>* arr = get_field_by_name(fname);
    cuda_arr_cmplx* arr = get_field_k_by_name[fname];
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> set_t(val, t_src);
}


/// Set fname to constant value at time index tlev=0
void slab_cuda::set_t(twodads::field_t fname, cuda::real_t val)
{
    //cuda_array<cuda::real_t >* arr = get_field_by_name(fname);
    cuda_arr_real* arr = get_field_by_name[fname];
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> set_t(val, 0);
}

/// advance all fields with multiple time levels
void slab_cuda :: advance()
{
    theta_hat.advance();
    theta_rhs_hat.advance();
    omega_hat.advance();
    omega_rhs_hat.advance();
}


/// Compute RHS from using time index t_src for dynamical fields omega_hat and theta_hat.
void slab_cuda :: rhs_fun(uint t_src)
{
    (this ->* theta_rhs_func)(t_src);
    (this ->* omega_rhs_func)(t_src);
}


// Update real fields theta, theta_x, theta_y, etc.
void slab_cuda::update_real_fields(uint tlev)
{
    //Compute theta, omega, strmf and respective spatial derivatives
    d_dx(twodads::f_theta_hat, twodads::f_theta_x_hat, tlev);
    d_dy(twodads::f_theta_hat, twodads::f_theta_y_hat, tlev);
    dft_c2r(twodads::f_theta_hat, twodads::f_theta, tlev);
    dft_c2r(twodads::f_theta_x_hat, twodads::f_theta_x, 0);
    dft_c2r(twodads::f_theta_y_hat, twodads::f_theta_y, 0);
    
    d_dx(twodads::f_omega_hat, twodads::f_omega_x_hat, tlev);
    d_dy(twodads::f_omega_hat, twodads::f_omega_y_hat, tlev);
    dft_c2r(twodads::f_omega_hat, twodads::f_omega, tlev);
    dft_c2r(twodads::f_omega_x_hat, twodads::f_omega_x, 0);
    dft_c2r(twodads::f_omega_y_hat, twodads::f_omega_y, 0);
    
    d_dx(twodads::f_strmf_hat, twodads::f_strmf_x_hat, 0);
    d_dy(twodads::f_strmf_hat, twodads::f_strmf_y_hat, 0);
    dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
    dft_c2r(twodads::f_strmf_x_hat, twodads::f_strmf_x, 0);
    dft_c2r(twodads::f_strmf_y_hat, twodads::f_strmf_y, 0);

    dft_c2r(twodads::f_theta_rhs_hat, twodads::f_theta_rhs, 1);
    dft_c2r(twodads::f_omega_rhs_hat, twodads::f_omega_rhs, 1);
}

// execute r2c DFT
void slab_cuda :: dft_r2c(twodads::field_t fname_r, twodads::field_k_t fname_c, uint t_src)
{
    cufftResult err;
    cuda_arr_real* arr_r = get_field_by_name[fname_r];
    cuda_arr_cmplx* arr_c = get_field_k_by_name[fname_c];
    err = cufftExecD2Z(plan_r2c, arr_r -> get_array_d(), (cufftDoubleComplex*) arr_c -> get_array_d(t_src));
    if (err != CUFFT_SUCCESS)
        throw;
}


// execute iDFT and normalize the resulting real field
void slab_cuda :: dft_c2r(twodads::field_k_t fname_c, twodads::field_t fname_r, uint t)
{
    cufftResult err;
    cuda_arr_cmplx* arr_c = get_field_k_by_name[fname_c];
    cuda_arr_real* arr_r = get_field_by_name[fname_r];
    err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) arr_c -> get_array_d(t), arr_r -> get_array_d());
    if (err != CUFFT_SUCCESS)
        throw;
    // Normalize
    arr_r -> normalize();
}


// print real field on terminal
void slab_cuda :: print_field(twodads::field_t field_name)
{
    cout << *get_field_by_name[field_name] << endl;
}

// print real field to ascii file
void slab_cuda :: print_field(twodads::field_t field_name, string file_name)
{
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_by_name[field_name] << "\n";
    output_file.close();
}



// print complex field on terminal, all time levels
void slab_cuda :: print_field(twodads::field_k_t field_name)
{
    cout << *get_field_k_by_name[field_name] << endl;
}


// print complex field to ascii file
void slab_cuda :: print_field(twodads::field_k_t field_name, string file_name)
{
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_k_by_name[field_name] << "\n";
    output_file.close();
}


/// Copy data from a cuda_array<cuda::real_t> to a cuda::real_t* buffer, tlev=0
void slab_cuda :: get_data(twodads::field_t fname, cuda::real_t* buffer)
{
    cuda_arr_real* arr = get_field_by_name[fname];
    for(uint t = 0; t < tlevs; t++)
        arr -> copy_device_to_host(buffer);
}

/// Update device data and return a pointer to requested array
cuda_arr_real* slab_cuda :: get_array_ptr(twodads::output_t fname)
{
    cuda_arr_real* arr = get_output_by_name[fname];
    arr -> copy_device_to_host();
    return arr;
}


/// Update device data and return a pointer to requested array
cuda_arr_real* slab_cuda :: get_array_ptr(twodads::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name[fname];
    arr -> copy_device_to_host();
    return arr;
}


void slab_cuda :: print_address()
{
    // Use this to test of memory is aligned between g++ and NVCC
    cout << "slab_cuda::dump_stiff_params()\n";
    cout << "config at " << (void*) &config << "\n";
    cout << "Nx at " << (void*) &Nx << "\n";
    cout << "My at " << (void*) &My << "\n";
    cout << "tlevs at " << (void*) &tlevs << "\n";
    cout << "plan_r2c at " << (void*) &plan_r2c << "\n";
    cout << "plan_c2r at " << (void*) &plan_c2r << "\n";
    cout << "theta at " << (void*) &theta << "\t";
    cout << "theta_x at " << (void*) &theta_x << "\t";
    cout << "theta_y at " << (void*) &theta_y << "\n";

    cout << "omega at " << (void*) &omega << "\t";
    cout << "omega_x at " << (void*) &omega_x << "\t";
    cout << "omega_y at " << (void*) &omega_y << "\n";

    cout << "strmf at " << (void*) &strmf << "\t";
    cout << "strmf_x at " << (void*) &strmf_x << "\t";
    cout << "strmf_y at " << (void*) &strmf_y << "\n";

    cout << "stiff_params at " << stiff_params << "\n";

    cout << "slab_cuda::slab_cuda()\n";
    cout << "\nsizeof(cuda::stiff_params_t) = " << sizeof(cuda::stiff_params_t); 
    cout << "\t at " << (void*) &stiff_params;
    cout << ": " << stiff_params;

    cout << "sizeof(cuda::slab_layuot_t) = " << sizeof(cuda::slab_layout_t);
    cout << "\t at " << (void*) &slab_layout;
    cout << ": " << slab_layout;
    cout << "\n\n";
}


void slab_cuda :: print_grids()
{
    cout << "Nx = " << Nx << ", My = " << My << "\t" << "Nx/2-1 = " << Nx/2-1 << endl;
    cout << "block_my_nx = (" << block_my_nx.x << ", " << block_my_nx.y << ")" << endl;
    cout << "grid_my_nx = (" << grid_my_nx.x << ", " << grid_my_nx.y << ")" << endl;
    cout << "block_my_nx21 = (" << block_my_nx21.x << ", " << block_my_nx21.y << ")" << endl;
    cout << "block_my_nx21 = (" << block_my_nx21.x << ", " << block_my_nx21.y << ")" << endl;
    cout << "grid_nx21_sec1 = (" << grid_nx21_sec1.x << ", " << grid_nx21_sec1.y << ")" << endl;
    cout << "grid_nx21_sec2 = (" << grid_nx21_sec2.x << ", " << grid_nx21_sec2.y << ")" << endl;
    cout << "grid_nx21_sec3 = (" << grid_nx21_sec3.x << ", " << grid_nx21_sec3.y << ")" << endl;
    cout << "grid_nx21_sec4 = (" << grid_nx21_sec4.x << ", " << grid_nx21_sec4.y << ")" << endl;
    cout << "grid_dx_half = (" << grid_dx_half.x << ", " << grid_dx_half.y << ")" << endl;
    cout << "grid_dx_single = (" << grid_dx_single.x << ", " << grid_dx_single.y << ")" << endl;
    cout << "grid_dy_half = (" << grid_dy_half.x << ", " << grid_dy_half.y << ")" << endl;
    cout << "grid_dy_single = (" << grid_dy_single.x << ", " << grid_dy_single.y << ")" << endl;
    cout << "grid_ky0 = (" << grid_ky0.x << ", " << grid_ky0.y << ")" << endl;
}


/// @brief RHS, set explicit part for theta equation to zero
/// @param t not used 
void slab_cuda :: theta_rhs_null(uint t)
{
    theta_rhs_hat = cuda::cmplx_t(0.0);
}

/// @brief RHS, set explicit part for omega equation to zero
/// @param t not used
void slab_cuda :: omega_rhs_null(uint t)
{
    omega_rhs_hat = cuda::cmplx_t(0.0);
}

/*****************************************************************************
 *
 * Function implementation
 *
 ****************************************************************************/


void slab_cuda :: enumerate(twodads::field_k_t f_name)
{
    get_field_k_by_name[f_name] -> enumerate_array(0);
}


void slab_cuda :: enumerate(twodads::field_t f_name)
{
    get_field_by_name[f_name] -> enumerate_array(0);
}


// Compute radial derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dy(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_arr_cmplx* arr_in = get_field_k_by_name[src_name];
    cuda_arr_cmplx* arr_out = get_field_k_by_name[dst_name];

    const uint Nx21 = Nx / 2 + 1;
    const double Ly = config.get_lengthy();

    d_d_dy_lo<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
    d_d_dy_mid<<<grid_dy_single, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
    d_d_dy_up<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: d_dy_enumerate(twodads::field_k_t f_name, uint t_src)
{
    cuda_arr_cmplx* arr = get_field_k_by_name[f_name];
    
    const uint Nx21 = Nx / 2 + 1;

    d_d_dy_lo_enumerate<<<grid_dy_half, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
    d_d_dy_mid_enumerate<<<grid_dy_single, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
    d_d_dy_up_enumerate<<<grid_dy_half, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Compute poloidal derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dx(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint tlev)
{
    cuda_arr_cmplx* arr_in = get_field_k_by_name[src_name];
    cuda_arr_cmplx* arr_out = get_field_k_by_name[dst_name];

    const uint Nx21 = Nx / 2 + 1;
    const double Lx = config.get_lengthx();

    d_d_dx_lo<<<grid_dx_half, block_nx21>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), My, Nx21, Lx);
    d_d_dx_up<<<grid_dx_single, block_nx21>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), My, Nx21, Lx);
#ifdef DEBUG
    gpuStatus();
#endif
}

// Compute poloidal derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dx_enumerate(twodads::field_k_t f_name, uint tlev)
{
    cuda_arr_cmplx* arr = get_field_k_by_name[f_name];

    const uint Nx21 = Nx / 2 + 1;
    cout << "d_dx_enumerate: Nx21 = " << Nx21 << ", My = " << My << endl;
    cout << "grid_dx_half = (" << grid_dx_half.x << ", " << grid_dx_half.y << ")" << endl;
    cout << "block_nx21 = (" << block_nx21.x << ", " << block_nx21.y << ")" << endl;
    d_d_dx_lo_enumerate<<<grid_dx_half, block_nx21>>>(arr -> get_array_d(tlev), My, Nx21);
    d_d_dx_up_enumerate<<<grid_dx_single, block_nx21>>>(arr -> get_array_d(tlev), My, Nx21);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_arr_cmplx* arr_in = get_field_k_by_name[src_name];
    cuda_arr_cmplx* arr_out = get_field_k_by_name[dst_name];

    const uint Nx21 = Nx / 2 + 1;
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

    d_inv_laplace_sec1<<<grid_nx21_sec1, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2<<<grid_nx21_sec2, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3<<<grid_nx21_sec3, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4<<<grid_nx21_sec4, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_zero<<<1, 1>>>(arr_out -> get_array_d(0));

#ifdef DEBUG
    gpuStatus();
#endif
}

// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace_enumerate(twodads::field_k_t f_name, uint t_src)
{
    cuda_arr_cmplx* arr = get_field_k_by_name[f_name];

    const uint Nx21 = Nx / 2 + 1;

    d_inv_laplace_sec1_enumerate<<<grid_nx21_sec1, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
    d_inv_laplace_sec2_enumerate<<<grid_nx21_sec2, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
    d_inv_laplace_sec3_enumerate<<<grid_nx21_sec3, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
    d_inv_laplace_sec4_enumerate<<<grid_nx21_sec4, block_nx21>>>(arr -> get_array_d(t_src), My, Nx21);
    d_inv_laplace_zero<<<1, 1>>>(arr -> get_array_d(0));
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: integrate_stiff(twodads::field_k_t fname, uint tlev)
{
    cuda_arr_cmplx* A = get_field_k_by_name[fname];
    cuda_arr_cmplx* A_rhs = rhs_array_map[fname];

    //d_integrate_stiff_debug<<<1, 1>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev, 4, 4);

    d_integrate_stiff_sec1<<<grid_nx21_sec1, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2<<<grid_nx21_sec2, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3<<<grid_nx21_sec3, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4<<<grid_nx21_sec4, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
#ifdef DEBUG
    gpuStatus();
#endif
}

/// @brief Integrate only modes with ky=0
void slab_cuda :: integrate_stiff_ky0(twodads::field_k_t fname, uint tlev)
{
    cuda_arr_cmplx* A = get_field_k_by_name[fname];
    cuda_arr_cmplx* A_rhs = rhs_array_map[fname];

    d_integrate_stiff_ky0<<<grid_dy_single, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
}


void slab_cuda :: integrate_stiff_enumerate(twodads::field_k_t fname, uint tlev)
{
    cuda_arr_cmplx* A = get_field_k_by_name[fname];
    cuda_arr_cmplx* A_rhs = rhs_array_map[fname];

    d_integrate_stiff_sec1_enumerate<<<grid_nx21_sec1, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2_enumerate<<<grid_nx21_sec2, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3_enumerate<<<grid_nx21_sec3, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4_enumerate<<<grid_nx21_sec4, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
#ifdef DEBUG
    gpuStatus();
#endif
} 

void slab_cuda :: integrate_stiff_debug(twodads::field_k_t fname, uint tlev, uint row, uint col)
{
    cuda_arr_cmplx* A = get_field_k_by_name[fname];
    cuda_arr_cmplx* A_rhs = rhs_array_map[fname];

    cout << "Debug information for stiffk\n";
    d_integrate_stiff_debug<<<1, 1>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev, row, col); 
    //d_integrate_stiff_sec1_enumerate<<<grid_nx21_sec1, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    //d_integrate_stiff_sec2_enumerate<<<grid_nx21_sec2, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    //d_integrate_stiff_sec3_enumerate<<<grid_nx21_sec3, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    //d_integrate_stiff_sec4_enumerate<<<grid_nx21_sec4, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
#ifdef DEBUG
    gpuStatus();
#endif
        
}


// Compute RHS for Navier stokes model, store result in time index 0 of theta_rhs_hat
void slab_cuda :: theta_rhs_ns(uint t_src)
{
    //cout << "theta_rhs_ns\n";
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


void slab_cuda :: theta_rhs_lin(uint t_src)
{
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);

#ifdef DEBUG
    gpuStatus();
#endif
}


///@brief Compute explicit part for Hasegawa-Wakatani model, store result in time index 0 of theta_rhs_hat
///@detailed $\mathcal{N}^0 = \left{n, \phi\right} + \mathcal{C} (\widetilde{phi} - n) - \phi_y$
void slab_cuda :: theta_rhs_hw(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    // Poisson bracket is on the RHS: dn/dt = {n, phi} + ...
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
    //theta_rhs_hat = 0.0;
    //gpuStatus(); 
    //d_theta_rhs_hw_debug<<<1, 1>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(0), theta_hat.get_array_d(t_src), strmf_y_hat.get_array_d(0), C, My, Nx / 2 + 1);
    d_theta_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(0), theta_hat.get_array_d(t_src), strmf_y_hat.get_array_d(0), C, My, Nx / 2 + 1);

#ifdef DEBUG
    gpuStatus();
#endif
}


/// @brief Explicit part for the MHW model
/// @detailed $\mathcal{N}^t = \left{n, \phi\right} - \mathcal{C} (\widetilde{phi} - \widetilde{n}) - \phi_y$
void slab_cuda :: theta_rhs_hwmod(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
    // Neglect ky=0 modes for in coupling term
    d_coupling_hwmod<<<grid_my_nx21, block_my_nx21>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
    theta_rhs_hat -= strmf_y_hat; 

#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: theta_rhs_log(uint t_src)
{
    //d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    //d_theta_rhs_log<<<grid_my_nx21, block_my_nx21>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), My, Nx);
    cout << "theta_rhs_log\n";
    d_theta_rhs_log<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: omega_rhs_ns(uint t_src)
{
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
#ifdef DEBUG
    gpuStatus();
#endif
}

/// @brief RHS for the Hasegawa-Wakatani model
/// @detailed RHS = {Omega, phi} + C(phi - n)
void slab_cuda :: omega_rhs_hw(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    //omega_rhs_hat = 0.0; 

    //d_omega_rhs_hw_debug<<<1, 1>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(0), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
    d_omega_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(0), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}

/// @brief RHS for modified Hasegawa-Wakatani model
/// @detailed: RHS = {Omega, phi} - \tilde{C(phi - n)}
void slab_cuda :: omega_rhs_hwmod(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    d_coupling_hwmod<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// @brief RHS for modified Hasegawa-Wakatani model, remove zonal flows
/// @detailed: RHS = \tilde{\Omega, phi} - C(phi, n)

void slab_cuda :: omega_rhs_hwzf(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    d_kill_ky0<<<grid_ky0, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), My, Nx / 2 + 1);
    d_omega_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda::omega_rhs_ic(uint t_src)
{
    // Convert model parameters to complex numbers
    const cuda::real_t ic = config.get_model_params(2); 
    const cuda::real_t sdiss = config.get_model_params(3);
    const cuda::real_t cfric = config.get_model_params(4);

    // Compute Poisson bracket in real space, use full grid/block
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    //dft_r2c(twodads::f_tmp, twodads::f_tmp_hat, 0);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
#ifdef DEBUG
    cout << "omega_rhs\n";
    cout << "ic = " << ic << ", sdiss = " << sdiss << ", cfric = " << cfric << "\n";
    cout << "grid = (" << omega_hat.get_grid().x << ", " << omega_hat.get_grid().y << "), block = (" << omega_hat.get_block().x << ", " << omega_hat.get_block().y << ")\n";
#endif //DEBUG
    //d_omega_ic_dummy<<<grid_my21_sec1, block_my21_sec1>>>(theta_y_hat.get_array_d(), strmf_hat.get_array_d(), omega_hat.get_array_d(0), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My / 2 + 1);
    d_omega_ic<<<grid_my_nx21, block_my_nx21>>>(theta_y_hat.get_array_d(0), strmf_hat.get_array_d(0), omega_hat.get_array_d(t_src), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), My, Nx / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}


// End of file slab_cuda.cu
