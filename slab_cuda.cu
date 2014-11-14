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
    if((col >= Nx21) || (row >= My / 2))
        return;

    out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * double(row));
}


__global__
void d_d_dy_lo_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;
    if((col >= Nx21) || (row >= My / 2))
        return;

    out[index] = cuda::cmplx_t(1000 + row, col);
}


// Frequencies: My/2
// These are stored in row My/2
__global__
void d_d_dy_mid(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = My / 2;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    // Return if we don't have an item to work on
    if(col >= Nx21)
        return;
    
    out[index] = cuda::cmplx_t(0.0, 0.0);
}


__global__
void d_d_dy_mid_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = My / 2;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;
    if(col >= Nx21)
        return;

    out[index] = cuda::cmplx_t(2000 + row, col);
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
    // Return if we don't have an item to work on
    if((col >= Nx21) || (row >= My))
        return;

    //cuda::cmplx_t factor(0.0, two_pi_L * (double(row) - double(My)));
    out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * (double(row) - double(My)));
}


__global__
void d_d_dy_up_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Ly)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if((col >= Nx21) || (row >= My))
        return;

    out[index] = cuda::cmplx_t(3000.0 + double(row) - double(My), col);
}


// x derivation
// Frequencies 0..Nx / 2, stored in cols 0..Nx/2-1
__global__
void d_d_dx_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Lx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    if ((col >= Nx21 - 1) || (row >= My))
        return;
    double two_pi_L = cuda::TWOPI / Lx;
    
    //(a + ib) * ik = -(b * k) + i(a * k)
    out[index] = in[index] * cuda::cmplx_t(0.0, two_pi_L * double(col));
}


__global__
void d_d_dx_lo_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Lx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;
    if ((col >= Nx21 - 1) || (row >= My))
        return;

    out[index] = cuda::cmplx_t(row, col);
}



// x derivation
// Frequencies Nx/2, stored in the last column. Set them to zero
__global__
void d_d_dx_up(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Lx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = Nx21 - 1;
    const uint index = row * Nx21 + col;

    if (col >= Nx21)
        return;

    out[index] = cuda::cmplx_t(0.0, 0.0);
}


__global__
void d_d_dx_up_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double Lx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = Nx21 - 1;
    const uint index = row * Nx21 + col;

    if (col >= Nx21)
        return;

    out[index] = cuda::cmplx_t(row, col);
}



// ky=0 modes are stored in the first row
__global__
void d_kill_ky0(cuda::cmplx_t* in, const uint My, const uint Nx21)
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = col;

    if (col >= Nx21)
        return;
    
    in[index] = cuda::cmplx_t(0.0, 0.0);
}


//
//
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
    if ((col > Nx21) || (row > My / 2))
        return;

    //cuda::cmplx_t factor(-cuda::FOURPIS * (double(col * col) * inv_Lx2 + double(row * row) * inv_Ly2), 0.0);
    out[idx] = in[idx] / cuda::cmplx_t(-cuda::FOURPIS * (double(col * col) * inv_Lx2 + double(row * row) * inv_Ly2), 0.0);
}


__global__
void d_inv_laplace_sec1_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((col >= Nx21) || (row >= My / 2))
        return;
    
    out[idx] = cuda::cmplx_t(1000 + row, col);
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
void d_inv_laplace_sec2_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((col >= Nx21) || (row >= My))
        return;

    out[idx] = cuda::cmplx_t(2000 + row - My, col);
}



// Pass Nx = Nx and My = My / 2 +1 for correct indexing
__global__
void d_inv_laplace_sec3(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 
    if (row > My / 2 + 1)
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            (double(row * row) * inv_Ly2 + double(col * col) * inv_Lx2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec3_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Ly2, const double inv_Lx2)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 
    if (row > My / 2 + 1)
        return;

    out[idx] = cuda::cmplx_t(3000 + row, col);
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
void d_inv_laplace_sec4_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, const uint My, const uint Nx21, const double inv_Lx2, const double inv_Ly2) 
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + My / 2 + 1;
    const uint col = Nx21 - 1;
    const uint idx = row * Nx21 + col; 

    if (row >= My)
        return;

    out[idx] = cuda::cmplx_t(4000 + row - My, col);
}


__global__
void d_inv_laplace_zero(cuda::cmplx_t* out)
{
    cuda::cmplx_t zero(0.0, 0.0);
    out[0] = zero;
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
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));
    

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

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = (cuda::real_t(row) - cuda::real_t(p.My)) * cuda::TWOPI / p.length_y;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

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
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

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
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

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
    cuda::real_t ky = 0.0;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div =  1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

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
void d_integrate_stiff_debug(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev, uint row, uint col, cuda::real_t kx, cuda::real_t ky)
{
    const uint idx = row * p.My + col;
    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    //cuda::real_t kx = cuda::TWOPI * cuda::real_t(row) / p.length_x;
    //cuda::real_t ky = cuda::TWOPI * cuda::real_t(row) / p.length_y;
    printf("delta_t = %f, diff = %f hv = %f, kx = %f, ky = %f\n", p.delta_t, p.diff, p.hv, kx, ky);
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky));
    cuda::cmplx_t result(0.0, 0.0);

    printf("\ttlev = %d, off_a = %d, off_b = %d\n", tlev, off_a, off_b);
    for(uint k = 1; k < tlev; k++)
    {
        printf("\ttlev=%d,k=%d\t %f * A[%d] + dt * %f * A_R[%d]\n", tlev, k, alpha[off_a - k], p.level - k, beta[off_b - k], p.level - 1 - k);
        printf("\ttlev=%d, k = %d\t sum_alpha += %f * (%f, %f)\n", tlev, k, alpha[off_a - k], (A[p.level -k][idx]).re(), (A[p.level -k][idx]).im());
        printf("\ttlev=%d, k = %d\t sum_beta+= %f * (%f, %f)\n", tlev, k, beta[off_b - k], (A_rhs[p.level - 1 - k][idx]).re(), (A_rhs[p.level - 1 - k][idx]).im());
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    result = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
    printf("\ttlev=%d, computing A[%d], gamma_0 = %f\n", tlev, p.level - tlev, alpha[(tlev - 2) * p.level]);
    printf("sum1_alpha = (%f, %f)\tsum1_beta = (%f, %f)\t", sum_alpha.re(), sum_alpha.im(), sum_beta.re(),  sum_beta.im());
    printf("temp_div = %f\n", temp_div); 
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev, idx, result.re(), result.im());
}


/*
 *
 * Kernels to compute non-linear operators
 *
 */


/// @brief Poisson brackt: {f, phi}
/// @param theta_x: f_x
/// @param theta_y: f_y
/// @param strmf_x: phi_x
/// @param strmf_y: phi_x
/// @param out: Field to store result in
/// @param Nx: Number of modes in x-direction (obs! field has Nx/2-1 columns)
/// @param My: Number of modes in y-direction
/// @detailed Poisson bracket is defined as d_y(f * phi_x) - d_x(f * phi_y)
/// @detailed When used to describe ExB advection the time derivative operator is partial f / partial t + {phi, f} = ...
/// @detailed In numerics, the Poisson bracket goes into the non-linear part of the time evolution equation:
/// @detailed df/dt + {phi, f} = .... goes over to partial f / partial t = {f, phi} + ...
/// @detailed
__global__
void d_pbracket(cuda::real_t* theta_x, cuda::real_t* theta_y, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t* out, const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    if ((row >= My) || (col >= Nx))
       return;
    out[idx] = theta_x[idx] * strmf_y[idx] - theta_y[idx] * strmf_x[idx];
}


// RHS for logarithmic density field:
// theta_x * strmf_x - theta_y * strmf_x + diff * (theta_x^2 + theta_y^2)
__global__
void d_theta_rhs_log(cuda::real_t* theta_x, cuda::real_t* theta_y, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t diff, cuda::real_t* tmp_arr, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((row >= My) || (col >= Nx21))
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
    theta_rhs_hat[idx] +=  ((strmf_hat[idx] - theta_hat[idx]) * C) - strmf_y_hat[idx];
}


__global__
void d_theta_rhs_hw_debug(cuda::cmplx_t* theta_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::cmplx_t* strmf_y_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint idx = 2;

    cuda::cmplx_t dummy = (theta_rhs_hat[idx] + (strmf_hat[idx] - theta_hat[idx]) * C - strmf_y_hat[idx]);
    printf("d_theta_rhs_hw_debug: initially: theta_rhs_hat[%d] = (%f, %f)\n", idx, theta_rhs_hat[idx].re(), theta_rhs_hat[idx].im());
    printf("                             --> theta_rhs_hat[%d] = (%f, %f)\tC = %f, theta_hat = (%f, %f), strmf_hat =(%f, %f), strmf_y_hat=(%f,%f)\n" ,
            idx, dummy.re(), dummy.im(), C, theta_hat[idx].re(), theta_hat[idx].im(), strmf_hat[idx].re(), strmf_hat[idx].im(), 
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

    cuda::cmplx_t foo((theta_y_hat[idx] * ic) + (strmf_hat[idx] * sdiss) + (omega_hat[idx] * cfric));
    out[idx] -= foo;
}


__global__
void d_omega_rhs_hw(cuda::cmplx_t* omega_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if ((row >= My) || (col >= Nx21))
       return;

    cuda::cmplx_t foo( (strmf_hat[idx] - theta_hat[idx]) * C);
    omega_rhs_hat[idx] += foo;
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


__global__
void d_omega_rhs_hw_debug(cuda::cmplx_t* omega_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint col = 2;
    const uint row = 0;
    const uint idx = row * My + col;

    cuda::cmplx_t dummy = ((strmf_hat[idx] - theta_hat[idx]) * C + omega_rhs_hat[idx]);
    //omega_rhs_hat[idx] = dummy;

    printf("d_omega_rhs_hw_debug: omega_rhs_hat[%d] = (%f, %f)\tC = %f, strmf_hat = (%f, %f), theta_hat =(%f, %f), strmf_y_hat=(%f,%f)\n" ,
            idx, dummy.re(), dummy.im(), C, strmf_hat[idx].re(), strmf_hat[idx].im(), theta_hat[idx].re(), theta_hat[idx].im());
}




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

    //* Setting RHS pointer for theta and omega corresponding to get_?????_rhs_type() from config */
    switch(config.get_theta_rhs_type())
    {
        case twodads::rhs_null:
            theta_rhs_fun = &slab_cuda::theta_rhs_null;
            break;
        case twodads::rhs_ns:
            theta_rhs_fun = &slab_cuda::theta_rhs_ns;
            break;
        case twodads::theta_rhs_lin:
            theta_rhs_fun = &slab_cuda::theta_rhs_lin;
            break;
        case twodads::theta_rhs_log:
            theta_rhs_fun = &slab_cuda::theta_rhs_log;
            break;
        case twodads::theta_rhs_hw:
            theta_rhs_fun = &slab_cuda::theta_rhs_hw;
            break;
        case twodads::theta_rhs_hwmod:
            theta_rhs_fun = &slab_cuda::theta_rhs_hwmod;
            break;
        case twodads::theta_rhs_ic:
            // Fall through
        case twodads::theta_rhs_NA:
            // Fall through
        default:
            //* Throw a name_error when the RHS for theta is not ot implemented yet*/
            string err_msg("Invalid RHS: RHS for theta not implemented yet\n");
            throw name_error(err_msg);
    }

    switch(config.get_omega_rhs_type())
    {
        case twodads::rhs_null:
            omega_rhs_fun = &slab_cuda::omega_rhs_null;
            break;
        case twodads::rhs_ns:
            omega_rhs_fun = &slab_cuda::omega_rhs_ns;
            break;
        case twodads::omega_rhs_ic:
            omega_rhs_fun = &slab_cuda::omega_rhs_ic;
            break;
        case twodads::omega_rhs_hw:
            omega_rhs_fun = &slab_cuda::omega_rhs_hw;
            break;
        case twodads::omega_rhs_hwmod:
            omega_rhs_fun = &slab_cuda::omega_rhs_hwmod;
            break;
        case twodads::omega_rhs_hwzf:
            omega_rhs_fun = &slab_cuda::omega_rhs_hwzf;
            break;
        case twodads::omega_rhs_NA:
            // Fall through
        default:
            //* Throw a name_error when the RHS for omega is not ot implemented yet*/
            string err_msg("Invalid RHS: RHS for omega not implemented yet\n");
            throw name_error(err_msg);
    
    }

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
/// @detailed After running initialize, the following things are set up:
/// @detailed theta, omega, strmf: Either by function specified through init_function string in 
/// @detailed input.ini or by taking the iDFT of their respective spectral field
/// @detailed theta_hat, omega_hat: non-zero values at last time index, tlev-1
/// @detailed omega_rhs_hat, theta_rhs_hat: computed as specified by omega/theta_rhs
/// @detailed non-zero value at last time index, tlev-2
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
            init_all_modes(&theta_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);


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


/// @brief Move data from time level t_src to t_dst
/// @param fname field name 
/// @param t_dst destination time index
/// @param t_src source time index
void slab_cuda :: move_t(twodads::field_k_t fname, uint t_dst, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr = get_field_by_name(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> move(t_dst, t_src);
}


/// @brief Copy data from time level t_src to t_dst
/// @param fname field name 
/// @param t_dst destination time index
/// @param t_src source time index
void slab_cuda :: copy_t(twodads::field_k_t fname, uint t_dst, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr = get_field_by_name(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> copy(t_dst, t_src);
}


/// @brief Set fname to a constant value at time index tlev
/// @param fname field name 
/// @param val constant complex number
/// @param t_src time index
void slab_cuda::set_t(twodads::field_k_t fname, cuda::cmplx_t val, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr = get_field_by_name(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> set_t(val, t_src);
}


/// @brief Set fname to constant value at time index tlev=0
/// @param fname field name
/// @param val constant real number
void slab_cuda::set_t(twodads::field_t fname, cuda::real_t val)
{
    cuda_array<cuda::real_t, cuda::real_t>* arr = get_field_by_name(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> set_t(val, 0);
}

/// @brief advance all fields with multiple time levels
void slab_cuda :: advance()
{
    theta_hat.advance();
    theta_rhs_hat.advance();
    omega_hat.advance();
    omega_rhs_hat.advance();
}


/// @brief Compute RHS from using time index t_src for dynamical fields omega_hat and theta_hat.
/// @param t_src The most current time level, only important for first transient time steps
void slab_cuda :: rhs_fun(uint t_src)
{
    (this ->* theta_rhs_fun)(t_src);
    (this ->* omega_rhs_fun)(t_src);
}


/// @brief Update real fields theta, theta_x, theta_y, etc.
/// @param tlev: The time level for theta_hat, omega_hat used as input for inverse DFT
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

/// @brief execute DFT
/// @param fname_r real field type
/// @param fname_c complex field type
/// @param t_src time index of complex field used as target for DFT
void slab_cuda :: dft_r2c(twodads::field_t fname_r, twodads::field_k_t fname_c, uint t_src)
{
    cufftResult err;
    cuda_array<cuda::real_t, cuda::real_t>* arr_r = get_field_by_name(fname_r);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_c = get_field_by_name(fname_c);
    err = cufftExecD2Z(plan_r2c, arr_r -> get_array_d(), (cufftDoubleComplex*) arr_c -> get_array_d(t_src));
    if (err != CUFFT_SUCCESS)
        throw;
}


/// @brief execute iDFT and normalize the resulting real field
/// @param fname_c complex field type
/// @param fname_r real field type
/// @param t time index of complex field used as source for iDFT
void slab_cuda :: dft_c2r(twodads::field_k_t fname_c, twodads::field_t fname_r, uint t)
{
    cufftResult err;
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_c= get_field_by_name(fname_c);
    cuda_array<cuda::real_t, cuda::real_t>* arr_r = get_field_by_name(fname_r);
    err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) arr_c -> get_array_d(t), arr_r -> get_array_d());
    if (err != CUFFT_SUCCESS)
        throw;
    // Normalize
    arr_r -> normalize();
}


/// @brief print real field on terminal
void slab_cuda :: print_field(twodads::field_t field_name)
{
    cuda_array<cuda::real_t, cuda::real_t>* field = get_field_by_name(field_name);
    cout << *field << "\n";
}

/// @brief print real field to ascii file
void slab_cuda :: print_field(twodads::field_t field_name, string file_name)
{
    cuda_arr_real* arr = get_field_by_name(field_name);
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *arr;
    output_file.close();
}



/// @brief print complex field on terminal, all time levels
/// @param field_name name of complex field
void slab_cuda :: print_field(twodads::field_k_t field_name)
{
    cuda_arr_cmplx* field = get_field_by_name(field_name);
    cout << *field << "\n";
}


/// @brief print complex field to ascii file
/// @param field_name: type of field to dump
/// @param file_name: name of the output file
void slab_cuda :: print_field(twodads::field_k_t field_name, string file_name)
{
    cuda_arr_cmplx* arr = get_field_by_name(field_name);
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *arr;
    output_file.close();
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
    cout << "block_my_nx = (" << block_my_nx.x << ", " << block_my_nx.y << ")\n";
    cout << "grid_my_nx = (" << grid_my_nx.x << ", " << grid_my_nx.y << ")\n";
    cout << "block_my_nx21 = (" << block_my_nx21.x << ", " << block_my_nx21.y << ")\n";
    cout << "block_my_nx21 = (" << block_my_nx21.x << ", " << block_my_nx21.y << ")\n";
    cout << "grid_nx21_sec1 = (" << grid_nx21_sec1.x << ", " << grid_nx21_sec1.y << ")\n";
    cout << "grid_nx21_sec2 = (" << grid_nx21_sec2.x << ", " << grid_nx21_sec2.y << ")\n";
    cout << "grid_nx21_sec3 = (" << grid_nx21_sec3.x << ", " << grid_nx21_sec3.y << ")\n";
    cout << "grid_nx21_sec4 = (" << grid_nx21_sec4.x << ", " << grid_nx21_sec4.y << ")\n";
    cout << "grid_dx_half = (" << grid_dx_half.x << ", " << grid_dx_half.y << ")\n";
    cout << "grid_dx_single = (" << grid_dx_single.x << ", " << grid_dx_single.y << ")\n";
    cout << "grid_ky0 = (" << grid_ky0.x << ", " << grid_ky0.y << ")\n";
}


/// @brief convert field type to internal pointer
/// @param field name of the real field
cuda_array<cuda::real_t, cuda::real_t>* slab_cuda :: get_field_by_name(twodads::field_t field)
{
    cuda_array<cuda::real_t, cuda::real_t>* result;
    switch(field)
    {
        case twodads::f_theta:
            result = &theta;
            break;
        case twodads::f_theta_x:
            result = &theta_x;
            break;
        case twodads::f_theta_y:
            result = &theta_y;
            break;
        case twodads::f_omega:
            result = &omega;
            break;
        case twodads::f_omega_x:
            result = &omega_x;
            break;
        case twodads::f_omega_y:
            result = &omega_y;
            break;
        case twodads::f_strmf:
            result = &strmf;
            break;
        case twodads::f_strmf_x:
            result = &strmf_x;
            break;
        case twodads::f_strmf_y:
            result = &strmf_y;
            break;
        case twodads::f_tmp:
            result = &tmp_array;
            break;
        case twodads::f_theta_rhs:
            result = &theta_rhs;
            break;
        case twodads::f_omega_rhs:
            result = &omega_rhs;
            break;
        default: 
            string err_str("Invalid field name\n");
            throw name_error(err_str);
    }
    return(result);
}


/// @brief convert field type to internal pointer
/// @param field name of the complex field
cuda_array<cuda::cmplx_t, cuda::real_t>* slab_cuda :: get_field_by_name(twodads::field_k_t field)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* result;
    switch(field)
    {
        case twodads::f_theta_hat:
            result = &theta_hat;
            break;
        case twodads::f_theta_x_hat:
            result = &theta_x_hat;
            break;
        case twodads::f_theta_y_hat:
            result = &theta_y_hat;
            break;
        case twodads::f_omega_hat:
            result = &omega_hat;
            break;
        case twodads::f_omega_x_hat:
            result = &omega_x_hat;
            break;
        case twodads::f_omega_y_hat:
            result = &omega_y_hat;
            break;
        case twodads::f_strmf_hat:
            result = &strmf_hat;
            break;
        case twodads::f_strmf_x_hat:
            result = &strmf_x_hat;
            break;
        case twodads::f_strmf_y_hat:
            result = &strmf_y_hat;
            break;
        case twodads::f_omega_rhs_hat:
            result = &omega_rhs_hat;
            break;
        case twodads::f_theta_rhs_hat:
            result = &theta_rhs_hat;
            break;
        case twodads::f_tmp_hat:
            result = &tmp_array_hat;
            break;
        default:
            string err_str("Invalid field name\n");
            throw name_error(err_str);
    }
    return (result);
}


/// @brief convert field type to internal pointer
/// @param fname name of the output field
/// @detailed This function is called by the output class. Copy device data to host
cuda_array<cuda::real_t, cuda::real_t>* slab_cuda :: get_field_by_name(twodads::output_t fname)
{
    cuda_array<cuda::real_t, cuda::real_t>* result;
    switch(fname)
    {
        case twodads::o_theta:
            theta.copy_device_to_host();
            result = &theta;
            break;
        case twodads::o_theta_x:
            theta_x.copy_device_to_host();
            result = &theta_x;
            break;
        case twodads::o_theta_y:
            theta_y.copy_device_to_host();
            result = &theta_y;
            break;
        case twodads::o_omega:
            omega.copy_device_to_host();
            result = &omega;
            break;
        case twodads::o_omega_x:
            omega_x.copy_device_to_host();
            result = &omega_x;
            break;
        case twodads::o_omega_y:
            omega_y.copy_device_to_host();
            result = &omega_y;
            break;
        case twodads::o_strmf:
            strmf.copy_device_to_host();
            result = &strmf;
            break;
        case twodads::o_strmf_x:
            strmf_x.copy_device_to_host();
            result = &strmf_x;
            break;
        case twodads::o_strmf_y:
            strmf_y.copy_device_to_host();
            result = &strmf_y;
            break;
        case twodads::o_theta_rhs:
            result = &theta_rhs;
            break;
        case twodads::o_omega_rhs:
            result = &omega_rhs;
            break;
        default:
            string err_str("get_field_by_name(twodads::output_t field): Invalid field name\n");
            throw name_error(err_str);
    }
    return(result);
}


/// @brief convert field type to internal pointer
/// @param fname name of the RHS field
cuda_array<cuda::cmplx_t, cuda::real_t>* slab_cuda :: get_rhs_by_name(twodads::field_k_t fname)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* result;
    switch(fname)
    {
        case twodads::f_theta_hat:
            result = &theta_rhs_hat;
            break;
        case twodads::f_omega_hat:
            result = &omega_rhs_hat;
            break;
        default: 
            string err_str("get_rhs_by_name(twodads::dyn_field_t fname): Invalid field name\n");
            throw name_error(err_str);
    }
    return(result);
}

/// @brief RHS, set explicit part for theta equation to zero
/// @param t not used 
void slab_cuda :: theta_rhs_null(uint t)
{
    CuCmplx<cuda::real_t> foobar(0.0, 0.0);
    theta_rhs_hat = foobar;
}

/// @brief RHS, set explicit part for omega equation to zero
/// @param t not used
void slab_cuda :: omega_rhs_null(uint t)
{
    CuCmplx<cuda::real_t> foobar(0.0, 0.0);
    omega_rhs_hat = foobar;
}

/*****************************************************************************
 *
 * Function implementation
 *
 ****************************************************************************/



// Compute radial derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dy(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);
    
    const uint Nx21 = Nx / 2 + 1;
    const double Ly = config.get_lengthy();

    d_d_dy_lo<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
    d_d_dy_mid<<<grid_dy_single, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
    d_d_dy_up<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: d_dy_enumerate(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);
    
    const uint Nx21 = Nx / 2 + 1;
    const double Ly = config.get_lengthy();

    d_d_dy_lo_enumerate<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
    d_d_dy_mid_enumerate<<<grid_dy_single, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
    d_d_dy_up_enumerate<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Compute poloidal derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dx(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint Nx21 = Nx / 2 + 1;
    const double Lx = config.get_lengthx();

    d_d_dx_lo<<<grid_dx_half, block_nx21>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), My, Nx21, Lx);
    d_d_dx_up<<<grid_dx_single, block_nx21>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), My, Nx21, Lx);
#ifdef DEBUG
    gpuStatus();
#endif
}

// Compute poloidal derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dx_enumerate(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint Nx21 = Nx / 2 + 1;
    double Lx = config.get_lengthx();

    d_d_dx_lo_enumerate<<<grid_dx_half, block_nx21>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), My, Nx21, Lx);
    d_d_dx_up_enumerate<<<grid_dx_single, block_nx21>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), My, Nx21, Lx);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

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
void slab_cuda :: inv_laplace_enumerate(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint Nx21 = Nx / 2 + 1;
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

    d_inv_laplace_sec1_enumerate<<<grid_nx21_sec1, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2_enumerate<<<grid_nx21_sec2, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3_enumerate<<<grid_nx21_sec3, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4_enumerate<<<grid_nx21_sec4, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, inv_Lx2, inv_Ly2);
    d_inv_laplace_zero<<<1, 1>>>(arr_out -> get_array_d(0));
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: integrate_stiff(twodads::field_k_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname); 
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname); 
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
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname);
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname);
    d_integrate_stiff_ky0<<<grid_dy_single, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
}


void slab_cuda :: integrate_stiff_enumerate(twodads::field_k_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname);
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname);
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
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname);
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname);

    // Compute kx and ky explicitly for mode number
    cuda::real_t kx = 0.0; 
    cuda::real_t ky = 0.0; 

    if(col < My / 2 + 1)
        ky = double(col) * 2.0 * cuda::PI / stiff_params.length_y;
    else
        return;

    if(row < Nx / 2 + 1)
        kx = double(row) * 2.0 * cuda::PI / stiff_params.length_x;
    else if ((row > Nx / 2) && row < Nx)
        kx = (double(row) - double(Nx)) * 2.0 * cuda::PI / stiff_params.length_x;
    else
        return;

    cout << "Debug information for stiffk\n";
    d_integrate_stiff_debug<<<1, 1>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev, row, col, kx, ky);
#ifdef DEBUG
    gpuStatus();
#endif
        
        //(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev, uint row, uint col, cuda::real_t kx, cuda::real_t ky)
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
#ifdef DEBUG
    gpuStatus();
#endif
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


///@brief Compute explicit part for Hasegawa-Wakatani model, store result in time index 0 of theta_rhs_hat
///@detailed $\mathcal{N}^0 = \left{n, \phi\right} + \mathcal{C} (\widetilde{phi} - n) - \phi_y$
void slab_cuda :: theta_rhs_hw(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    // Poisson bracket is on the RHS: dn/dt = {n, phi} + ...
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
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
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    d_theta_rhs_log<<<grid_my_nx21, block_my_nx21>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), My, Nx);
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
/// @detailed RHS = {Omega, phi} - C(phi - n)
void slab_cuda :: omega_rhs_hw(uint t_src)
{
    const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    d_omega_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
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
    d_kill_ky0<<<grid_ky0, block_my_nx21>>>(tmp_array.get_array_d(), My, Nx / 2 + 1);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    d_omega_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda::omega_rhs_ic(uint t_src)
{
    // Compute Poisson bracket in real space, use full grid/block
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_tmp_hat, 0);
    // Convert model parameters to complex numbers
    const cuda::real_t ic = config.get_model_params(2); 
    const cuda::real_t sdiss = config.get_model_params(3);
    const cuda::real_t cfric = config.get_model_params(4);
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
