#include "include/error.h"
#include "include/slab_cuda.h"
#include <algorithm>

/// Standard constructor for slab_cuda

using namespace std;
/*****************************************************************************
 *
 * Kernel implementation
 *
 ****************************************************************************/

///d/dx: Frequencies 0 .. N/2 - 1
__global__
void d_d_dx_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;

    double two_pi_L = cuda::TWOPI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    // (a + ib) * (0.0 + ik) = -(b * k) + i(a * k)
    cuda::cmplx_t factor(0.0, two_pi_L * double(row));
    out[index] = in[index] * factor;
}


__global__
void d_d_dx_lo_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
   const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;
    if((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(100000 + 1 * 1000 * col + row, row);
    out[index] = factor;
}


// Frequencies: Nx/2
__global__
void d_d_dx_mid(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = Nx / 2;
    const uint index = row * My + col;

    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    cuda::cmplx_t res(0.0, 0.0);
    out[index] = res;
}


__global__
void d_d_dx_mid_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = Nx / 2;
    const uint index = row * My + col;
    if((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(200000 + 1000 * col + row);
    out[index] = factor;
}



// Frequencies: Nx/2 + 1 ... Nx - 1
__global__
void d_d_dx_up(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint index = row * My + col;

    double two_pi_L = cuda::TWOPI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    //(a + ib) * (0.0 + ik) = -(b * k) + i * (a * k)
    cuda::cmplx_t factor(0.0, two_pi_L * (double(row) - double(Nx)));
    out[index] = in[index] * factor;
}


__global__
void d_d_dx_up_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint index = row * My + col;
    if((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(300000 + 1000 * col + double(row) - double(Nx), double(row) - double(Nx));
    out[index] = factor;
}


// Frequencies 0..My / 2
__global__
void d_d_dy_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Ly)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;

    if ((col > My - 2) || (row >= Nx))
        return;
    double two_pi_L = cuda::TWOPI / Ly;
    
    //(a + ib) * ik = -(b * k) + i(a * k)
    cuda::cmplx_t factor(0.0, two_pi_L * double(col));
    out[index] = factor * in[index];
}


__global__
void d_d_dy_lo_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Ly)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;
    if ((col > My - 2) || (row >= Nx))
        return;

    cuda::cmplx_t factor(100000 + 1000 * col + row, col);
    out[index] = factor;
}



__global__
void d_d_dy_up(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Ly)
{
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;

    if (col > Nx + 1)
        return;
    cuda::cmplx_t factor(0.0, 0.0);
    out[index] = factor;
}


__global__
void d_d_dy_up_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Ly)
{
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;
    //uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (col > Nx + 1)
        return;

    //index = (index + 1) * My - 1;
    cuda::cmplx_t factor(double(200000 + 1000 * col) + 0.0, 0.0);
    out[index] = factor;
}




//
//
// invert two dimensional laplace equation.
// In spectral space, 
//                              / 4 pi^2 ((kx/Lx)^2 + (ky/Ly)^2 )  for kx, ky  <= N/2
// phi(kx, ky) = omega(kx, ky)  
//                              / 4 pi^2 (((kx-Nx)/Lx)^2 + (ky/Ly)^2) for kx > N/2 and ky <= N/2
//
// and phi(0,0) = 0 (to avoid division by zero)
// Divide into 4 sectors:
//
//            My/2    1 (last element)
//         ^<------>|------|
//  Nx/2+1 |        |      |
//         |   I    | III  |
//         |        |      |
//         v        |      |
//         =================
//         ^        |      |
//         |        |      |
//  Nx/2-1 |  II    |  IV  |   
//         |        |      |
//         v<------>|------|
//           My/2      1
//
// 
// sector I    : kx <= Nx/2, ky <= My/2  BS = (1, cuda_blockdim_my), GS = (Nx/2+1, My / (2 * cuda_blockdim_my)
// sector II   : kx > Nx/2, ky <= My/2   BS = (1, cuda_blockdim_my), GS = (Nx/2-1, My / (2 * cuda_blockdim_my)
// sector III  : kx <= Nx/2, ky = My/2   BS = cuda_blockdim_nx, GS = ^(Nx / 2 + 1) / cuda_blockdim_nx^ (round up, thread returns if out of bounds)
// sector IV   : kx > Nx/2, ky = My/2    BS = cuda_blockdim_nx, GS = ^(Nx / 2 - 1) / cuda_blockdim_nx^ (round up, thread returns if out of bounds)
//
// Pro: wavenumbers can be computed from index without if-else blocks
// Con: Diverging memory access

__global__
void d_inv_laplace_sec1(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec1_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;
    
    cuda::cmplx_t factor(100000 + 1000 * col + row, 1000 * col + row);
    out[idx] = factor;
}


__global__
void d_inv_laplace_sec2(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            ((double(row) - double(Nx)) * (double(row) - double(Nx))) * inv_Lx2 +
            (double(col * col) * inv_Ly2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec2_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(200000.0 + 1000.0 * double(col) + double(row), 1000.0 * double(col) + (double(row) - double(Nx)) ); 
    out[idx] = factor;
}



// Pass Nx = Nx and My = My / 2 +1 for correct indexing
__global__
void d_inv_laplace_sec3(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col; 
    if (row > Nx / 2 + 1)
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec3_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col; 
    if (row > Nx / 2 + 1)
        return;

    cuda::cmplx_t factor(300000 + 1000 * col + row, 1000 * col + row);
    out[idx] = factor;
}


// Pass Nx = Nx and My = My / 2 + 1 for correct indexing
__global__
void d_inv_laplace_sec4(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint col = My - 1;
    const uint idx = row * My + col; 

    if (row >= Nx)
        return;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            ((double(row) - double(Nx)) * (double(row) - double(Nx)) * inv_Lx2 + double(col * col) * inv_Ly2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec4_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = row * My + col; 

    if (row >= Nx)
        return;

    cuda::cmplx_t factor(double(400000 + 1000 * col + row), 1000. * double(col) + (double(row) - double(Nx)) ); 
    out[idx] = factor;
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
 *
 * struct stiff_params{
 *  real_t delta_t;
 *  real_t length_x; 
 *  real_t length_y;
 *  real_t diff;
 *  real_t hv;
 *  uint Nx;
 *  uint My;
 *  uint tlev;  
 */




__global__
void d_integrate_stiff_sec1_enumerate(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev) 

{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.My + col;
    if ((col >= p.My) || (row >= p.Nx))
        return;

    cuda::cmplx_t value(100000 + 1000 * col + row, 1000 * col + row);
    A_rhs[p.level - tlev][idx] = value;
}



__global__
void d_integrate_stiff_sec1(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev) 

{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.My + col;
    if ((col >= p.My) || (row >= p.Nx))
        return;

    unsigned int off_a = (tlev - 2) * p.level + tlev;
    unsigned int off_b = (tlev - 2) * (p.level - 1) + tlev- 1;
    cuda::real_t kx = cuda::real_t(row) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(col) * cuda::TWOPI / p.length_y;
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));
    

    // Add contribution from explicit / implicit parts
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
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + p.Nx / 2 +  1;
    const uint idx = row * p.My + col;
    if ((col >= p.My) || (row >= p.Nx))
        return;

    cuda::cmplx_t value(200000.0 + 1000.0 * double(col) + double(row), 1000.0 * double(col) + (double(row) - double(p.Nx)));
    A_rhs[p.level - tlev][idx] = value;
}

__global__
void d_integrate_stiff_sec2(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + p.Nx / 2 + 1;
    const uint idx = row * p.My + col;
    if ((col >= p.My) || (row >= p.Nx))
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = (cuda::real_t(row) - cuda::real_t(p.Nx)) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(col) * cuda::TWOPI / p.length_y;
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
    const uint col = p.My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.My + col; 
    if(row > p.Nx / 2 + 1)
        return;

    cuda::cmplx_t value(300000 + 1000 * col + row, 1000 * col + row);
    A_rhs[p.level - tlev][idx] = value;
}

__global__
void d_integrate_stiff_sec3(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = p.My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * p.My + col; 
    if(row > p.Nx / 2 + 1)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(row) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(col) * cuda::TWOPI/ p.length_y;
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
    const uint col = p.My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + p.Nx / 2 + 1;
    const uint idx = row * p.My + col; 
    if (row >= p.Nx)
        return;

    cuda::cmplx_t value(double(400000 + 1000 * col + row), 1000. * col + (double(row) - double(p.Nx)) );
    A_rhs[p.level - tlev][idx] = value;
}

__global__
void d_integrate_stiff_sec4(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = p.My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + p.Nx / 2 + 1;
    const uint idx = row * p.My + col; 
    if (row >= p.Nx)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = (cuda::real_t(row) - cuda::real_t(p.Nx)) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(col) * cuda::TWOPI / p.length_y;
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
/// @param Nx: Number of modes in x-direction
/// @param My: Number of modes in y-direction
/// @detailed Poisson bracket is defined as d_y(f * phi_x) - d_x(f * phi_y)
/// @detailed When used to describe ExB advection the time derivative operator is partial f / partial t + {phi, f} = ...
/// @detailed In numerics, the Poisson bracket goes into the non-linear part of the time evolution equation:
/// @detailed df/dt + {phi, f} = .... goes over to partial f / partial t = {f, phi} + ...
/// @detailed
__global__
void d_pbracket(cuda::real_t* theta_x, cuda::real_t* theta_y, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t* out, uint Nx, uint My)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;

    if ((col >= My) || (row >= Nx))
       return;
    out[idx] = theta_x[idx] * strmf_y[idx] - theta_y[idx] * strmf_x[idx];
}


// RHS for logarithmic density field:
// theta_x * strmf_x - theta_y * strmf_x + diff * (theta_x^2 + theta_y^2)
__global__
void d_theta_rhs_log(cuda::real_t* theta_x, cuda::real_t* theta_y, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t diff, cuda::real_t* tmp_arr, uint Nx, uint My)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
       return;

    tmp_arr[idx] = theta_x[idx] * strmf_y[idx] - theta_y[idx] * strmf_x[idx] + diff * (theta_x[idx] * theta_x[idx] + theta_y[idx] * theta_y[idx]);
}


__global__
void d_theta_rhs_hw(cuda::cmplx_t* theta_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::cmplx_t* strmf_y_hat, cuda::real_t C, uint Nx, uint My)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;
    //theta_rhs_hat[idx].x += C * (strmf_hat[idx].x - theta_hat[idx].x) - strmf_y_hat[idx].x;
    //theta_rhs_hat[idx].y += C * (strmf_hat[idx].y - theta_hat[idx].y) - strmf_y_hat[idx].y;
    theta_rhs_hat[idx] +=  ((strmf_hat[idx] - theta_hat[idx]) * C) - strmf_y_hat[idx];
}


__global__
void d_theta_rhs_hw_debug(cuda::cmplx_t* theta_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::cmplx_t* strmf_y_hat, cuda::real_t C, uint Nx, uint My)
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
void d_omega_ic_sec1(cuda::cmplx_t* theta_y_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* omega_hat, cuda::real_t ic, cuda::real_t sdiss, cuda::real_t cfric, cuda::cmplx_t* out, uint Nx, uint My)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
       return;

    cuda::cmplx_t foo((theta_y_hat[idx] * ic) + (strmf_hat[idx] * sdiss) + (omega_hat[idx] * cfric));
    out[idx] -= foo;
}


__global__
void d_omega_rhs_hw(cuda::cmplx_t* omega_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::real_t C, uint Nx, uint My)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
       return;

    cuda::cmplx_t foo( (strmf_hat[idx] - theta_hat[idx]) * C);
    omega_rhs_hat[idx] += foo;
}


__global__
void d_coupling_hwmod(cuda::cmplx_t* rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::real_t C, uint Nx, uint My)
{
    // Start columns with offset 1, this skips all ky=0 modes
    const uint col = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t dummy = (strmf_hat[idx] - theta_hat[idx]) * C;
    rhs_hat[idx] += dummy;
}


__global__
void d_omega_rhs_hw_debug(cuda::cmplx_t* omega_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::real_t C, uint Nx, uint My)
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
    Nx(my_config.get_nx()),
    My(my_config.get_my()),
    tlevs(my_config.get_tlevs()),
    theta(1, Nx, My), theta_x(1, Nx, My), theta_y(1, Nx, My),
    omega(1, Nx, My), omega_x(1, Nx, My), omega_y(1, Nx, My),
    strmf(1, Nx, My), strmf_x(1, Nx, My), strmf_y(1, Nx, My),
    tmp_array(1, Nx, My), 
    theta_rhs(1, Nx, My), omega_rhs(1, Nx, My),
    theta_hat(tlevs, Nx, My / 2 + 1), theta_x_hat(1, Nx, My / 2 + 1), theta_y_hat(1, Nx, My / 2 + 1),
    omega_hat(tlevs, Nx, My / 2 + 1), omega_x_hat(1, Nx, My / 2 + 1), omega_y_hat(1, Nx, My / 2 + 1),
    strmf_hat(1, Nx, My / 2 + 1), strmf_x_hat(1, Nx, My / 2 + 1), strmf_y_hat(1, Nx, My / 2 + 1),
    tmp_array_hat(1, Nx, My / 2 + 1), 
    theta_rhs_hat(tlevs - 1, Nx, My / 2 + 1),
    omega_rhs_hat(tlevs - 1, Nx, My / 2 + 1),
    dft_is_initialized(init_dft()),
    stiff_params(config.get_deltat(), config.get_lengthx(), config.get_lengthy(), config.get_model_params(0),
            config.get_model_params(1), Nx, My / 2 + 1, tlevs),
    slab_layout(config.get_xleft(), config.get_deltax(), config.get_ylow(), config.get_deltay(), Nx, My),
    block_nx_my(theta.get_block()),
    grid_nx_my(theta.get_grid()),
    block_my21_sec1(theta_hat.get_block()),
    grid_my21_sec1(theta_hat.get_grid()),
    block_my21_sec2(dim3(cuda::cuda_blockdim_nx)),
    grid_my21_sec2(dim3((Nx + cuda::cuda_blockdim_nx - 1) / cuda::cuda_blockdim_nx)),
    grid_dx_half(dim3(Nx / 2, theta_hat.get_grid().y)),
    grid_dx_single(dim3(1, theta_hat.get_grid().y))
{

    cudaDeviceSynchronize();
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
        case twodads::theta_rhs_NA:
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
        case twodads::omega_rhs_NA:
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

    //* Initialize block and grid sizes for inv_lapl and integrate_stiff */
    uint bs_y_sec12 = min(cuda::cuda_blockdim_my, My / 2);
    uint gs_y_sec12 = My / (2 * bs_y_sec12);
    uint num_blocks_sec3 = ((Nx / 2 + 1) + (cuda::cuda_blockdim_nx - 1)) / cuda::cuda_blockdim_nx;
    uint num_blocks_sec4 = ((Nx / 2 - 1) + (cuda::cuda_blockdim_nx - 1)) / cuda::cuda_blockdim_nx;

    block_sec12 = dim3(1, bs_y_sec12);
    grid_sec1 = dim3(Nx / 2 + 1, gs_y_sec12);
    grid_sec2 = dim3(Nx / 2 - 1, gs_y_sec12);
    block_sec3 = dim3(cuda::cuda_blockdim_nx);
    block_sec4 = dim3(cuda::cuda_blockdim_nx);
    grid_sec3 = dim3(num_blocks_sec3);
    grid_sec4 = dim3(num_blocks_sec4);

//#ifdef DEBUG
//    cout << "block_my21_sec1 = (" << block_my21_sec1.x << ", " << block_my21_sec1.y << ")\t";
//    cout << "grid_my21_sec1 = (" << grid_my21_sec1.x << ", " << grid_my21_sec1.y << ")\n";
//    cout << "block_my21_sec2 = (" << block_my21_sec2.x << ", " << block_my21_sec2.y << ")\t";
//    cout << "grid_my21_sec2 = (" << grid_my21_sec2.x << ", " << grid_my21_sec2.y << ")\n";
//    cout << "block_sec12 = (" << block_sec12.x << ", " << block_sec12.y << ")\t";
//    cout << "block_sec3 = (" << block_sec3.x << ", " << block_sec3.y << ")\t";
//    cout << "block_sec4 = (" << block_sec4.x << ", " << block_sec4.y << ")\t";
//    cout << "grid_sec1 = (" << grid_sec1.x << ", " << grid_sec1.y << ")\n";
//    cout << "grid_sec2 = (" << grid_sec2.x << ", " << grid_sec2.y << ")\n";
//    cout << "grid_sec3 = (" << grid_sec3.x << ", " << grid_sec3.y << ")\n";
//    cout << "grid_sec4 = (" << grid_sec4.x << ", " << grid_sec4.y << ")\n";
//#endif //DEBUG
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
    cout << "slab_cuda :: initialize()\n";
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
            cout << "Initializing not available\n";
            break;

        case twodads::init_theta_gaussian:
            cout << "Initizlizing theta gaussian\n";
            init_gaussian(&theta, config.get_initc(), slab_layout, config.get_log_theta());
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_both_gaussian:
            cout << "Initializing theta, omega as aguasian\n";
            init_gaussian(&omega, config.get_initc(), slab_layout, 0);
            dft_r2c(twodads::f_omega, twodads::f_omega_hat, config.get_tlevs() - 1);

            init_gaussian(&theta, config.get_initc(), slab_layout, config.get_log_theta());
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_theta_mode:
            cout << "Initializing single mode for theta\n";
            init_mode(&theta_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);
            dft_c2r(twodads::f_theta_hat, twodads::f_theta, config.get_tlevs() - 1);

            break;

        case twodads::init_omega_mode:
            cout << "Initializing single mode for omega\n";
            init_mode(&omega_hat, config.get_initc(), slab_layout, config.get_tlevs()  - 1);

            // Compute stream function and spatial derivatives for omega, and phi
            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);
            break;

        case twodads::init_both_mode:
            cout << "Initializing single modes for theta and omega\n";
            init_mode(&theta_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);
            init_mode(&omega_hat, config.get_initc(), slab_layout, config.get_tlevs() - 1);

            inv_laplace(twodads::f_omega_hat, twodads::f_strmf_hat, config.get_tlevs() - 1);
            dft_c2r(twodads::f_strmf_hat, twodads::f_strmf, 0);

            break;

        case twodads::init_theta_sine:
            cout << "init_theta_sine\n";
            init_simple_sine(&theta, config.get_initc(), slab_layout);
            dft_r2c(twodads::f_theta, twodads::f_theta_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_omega_sine:
            cout << "init_omega_sine\n";
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
            cout << "init_file\n";
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


/// @brief dump real field on terminal
void slab_cuda :: dump_field(twodads::field_t field_name)
{
    cuda_array<cuda::real_t, cuda::real_t>* field = get_field_by_name(field_name);
    cout << *field << "\n";
}

/// @brief dump real field to ascii file
void slab_cuda :: dump_field(twodads::field_t field_name, string file_name)
{
    cuda_arr_real* arr = get_field_by_name(field_name);
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *arr;
    output_file.close();
}



/// @brief dump complex field on terminal, all time levels
/// @param field_name name of complex field
void slab_cuda :: dump_field(twodads::field_k_t field_name)
{
    cuda_arr_cmplx* field = get_field_by_name(field_name);
    cout << *field << "\n";
}


/// @brief dump complex field to ascii file
/// @param field_name: type of field to dump
/// @param file_name: name of the output file
void slab_cuda :: dump_field(twodads::field_k_t field_name, string file_name)
{
    cuda_arr_cmplx* arr = get_field_by_name(field_name);
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *arr;
    output_file.close();
}



/// @brief write full output to output.h5
/// @param time real number to be written as attribute of the datasets
/*
void slab_cuda :: write_output(twodads::real_t time)
{
#ifdef DEBUG
    cudaDeviceSynchronize();
    cout << "Writing output, time = " << time << "\n";
#endif //DEBUG
    for(auto field_name : config.get_output())
        slab_output.surface(field_name, get_field_by_name(field_name), time);
    slab_output.output_counter++;
}
*/

void slab_cuda ::dump_stiff_params()
{
    // Use this to test of memory is aligned between g++ and NVCC
    cout << "slab_cuda::dump_stiff_params()\n";
    cout << "config at " << (void*) &config << "\n";
    cout << "Nx at " << (void*) &Nx << "\n";
    cout << "My at " << (void*) &My << "\n";
    cout << "tlevs at " << (void*) &tlevs << "\n";
    cout << "plan_r2c at " << (void*) &plan_r2c << "\n";
    cout << "plan_c2r at " << (void*) &plan_c2r << "\n";
    cout << "theta at " << (void*) &theta << "\n";
    cout << "theta_x at " << (void*) &theta_x << "\n";
    cout << "theta_y at " << (void*) &theta_y << "\n";
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
/// @param fname name of the dynamic field
/*
cuda_array<cuda::cmplx_t, cuda::real_t>* slab_cuda :: get_field_by_name(twodads::dyn_field_t fname)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* result;
    switch(fname)
    {
        case twodads::d_theta:
            result = &theta_hat;
            break;
        case twodads::d_omega:
            result = &omega_hat;
            break;
        default: 
            string err_str("get_field_by_name(twodads::dyn_field_t): Invalid field name\n");
            throw name_error(err_str);
    }
    return(result);
}
*/

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
void slab_cuda :: d_dx(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);
    
    const uint my21 = My / 2 + 1;
    double Lx = config.get_deltax() * double(Nx);

    d_d_dx_lo<<<grid_dx_half, block_my21_sec1>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, Lx);
    d_d_dx_mid<<<grid_dx_single, block_my21_sec1>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, Lx);
    d_d_dx_up<<<grid_dx_half, block_my21_sec1>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, Lx);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: d_dx_enumerate(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);
    
    const uint my21 = My / 2 + 1;
    double Lx = config.get_deltax() * double(Nx);

    d_d_dx_lo_enumerate<<<grid_dx_half, block_my21_sec1>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, Lx);
    d_d_dx_mid_enumerate<<<grid_dx_single, block_my21_sec1>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, Lx);
    d_d_dx_up_enumerate<<<grid_dx_half, block_my21_sec1>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, Lx);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Compute poloidal derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dy(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint my21 = My / 2 + 1;
    double Ly = config.get_lengthy();

    d_d_dy_lo<<<arr_in -> get_grid(), arr_out -> get_block()>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), Nx, my21, Ly);
    d_d_dy_up<<<grid_my21_sec2, block_my21_sec2>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), Nx, my21, Ly);
#ifdef DEBUG
    gpuStatus();
#endif
}

// Compute poloidal derivative from src_name using time index t_src, store in dst_name, time index 0
void slab_cuda :: d_dy_enumerate(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint my21 = My / 2 + 1;
    double Ly = config.get_lengthy();

    d_d_dy_lo_enumerate<<<arr_in -> get_grid(), arr_out -> get_block()>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), Nx, my21, Ly);
    d_d_dy_up_enumerate<<<grid_my21_sec2, block_my21_sec2>>>(arr_in -> get_array_d(tlev), arr_out -> get_array_d(0), Nx, my21, Ly);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint my21 = My / 2 + 1;
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

//#ifdef DEBUG
//    cout << "slab_cuda::inv_laplace(...)\n";
//    cout << "block_sec12 = (" << block_sec12.x << ", " << block_sec12.y << ")\t";
//    cout << "block_sec3 = (" << block_sec3.x << ", " << block_sec3.y << ")\t";
//    cout << "block_sec4 = (" << block_sec4.x << ", " << block_sec4.y << ")\t";
//    cout << "grid_sec1 = (" << grid_sec1.x << ", " << grid_sec1.y << ")\n";
//    cout << "grid_sec2 = (" << grid_sec2.x << ", " << grid_sec2.y << ")\n";
//    cout << "grid_sec3 = (" << grid_sec3.x << ", " << grid_sec3.y << ")\n";
//    cout << "grid_sec4 = (" << grid_sec4.x << ", " << grid_sec4.y << ")\n";
//#endif //DEBUG

    d_inv_laplace_sec1<<<grid_sec1, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2<<<grid_sec2, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3<<<grid_sec3, block_sec3>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4<<<grid_sec4, block_sec4>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
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

    const uint my21 = My / 2 + 1;
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

    d_inv_laplace_sec1_enumerate<<<grid_sec1, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2_enumerate<<<grid_sec2, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3_enumerate<<<grid_sec3, block_sec3>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4_enumerate<<<grid_sec4, block_sec4>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_zero<<<1, 1>>>(arr_out -> get_array_d(0));
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: integrate_stiff(twodads::field_k_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname); 
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname); 
    //d_integrate_stiff_debug<<<1, 1>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec1<<<grid_sec1, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2<<<grid_sec2, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3<<<grid_sec3, block_sec3>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4<<<grid_sec4, block_sec4>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: integrate_stiff_enumerate(twodads::field_k_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname);
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname);
    cout << "integrate_stiff_enumerate\n";
    d_integrate_stiff_sec1_enumerate<<<grid_sec1, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2_enumerate<<<grid_sec2, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3_enumerate<<<grid_sec3, block_sec3>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4_enumerate<<<grid_sec4, block_sec4>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
#ifdef DEBUG
    gpuStatus();
#endif
    cudaDeviceSynchronize();
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
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


void slab_cuda :: theta_rhs_lin(uint t_src)
{
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


///@brief Compute explicit part for Hasegawa-Wakatani model, store result in time index 0 of theta_rhs_hat
///@detailed $\mathcal{N}^0 = \left{n, \phi\right} + \mathcal{C} (\widetilde{phi} - n) - \phi_y$
void slab_cuda :: theta_rhs_hw(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    // Poisson bracket is on the RHS: dn/dt = {n, phi} + ...
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
    d_theta_rhs_hw<<<grid_my21_sec1, block_my21_sec1>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(0), theta_hat.get_array_d(t_src), strmf_y_hat.get_array_d(0), C, Nx, My / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// @brief Explicit part for the MHW model
/// @detailed $\mathcal{N}^t = \left{n, \phi\right} - \mathcal{C} (\widetilde{phi} - \widetilde{n}) - \phi_y$
void slab_cuda :: theta_rhs_hwmod(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
    // Neglect ky=0 modes for in coupling term
    d_coupling_hwmod<<<grid_my21_sec1, block_my21_sec1>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, Nx, My / 2 + 1);
    theta_rhs_hat -= strmf_y_hat; 
#ifdef DEBUG
    gpuStatus();
#endif
}

void slab_cuda :: theta_rhs_log(uint t_src)
{
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    d_theta_rhs_log<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: omega_rhs_ns(uint t_src)
{
    d_pbracket<<<grid_nx_my, block_nx_my>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
#ifdef DEBUG
    gpuStatus();
#endif
}

/// @brief RHS for the Hasegawa-Wakatani model
/// @detailed RHS = {Omega, phi} - C(phi - n)
void slab_cuda :: omega_rhs_hw(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_nx_my, block_nx_my>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    d_omega_rhs_hw<<<grid_my21_sec1, block_my21_sec1>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, Nx, My / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}


void slab_cuda :: omega_rhs_hwmod(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_nx_my, block_nx_my>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    d_coupling_hwmod<<<grid_my21_sec1, block_my21_sec1>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, Nx, My / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}

void slab_cuda::omega_rhs_ic(uint t_src)
{
    // Compute Poisson bracket in real space, use full grid/block
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_tmp_hat, 0);
    // Convert model parameters to complex numbers
    cuda::real_t ic = config.get_model_params(2); 
    cuda::real_t sdiss = config.get_model_params(3);
    cuda::real_t cfric = config.get_model_params(4);
#ifdef DEBUG
    cout << "omega_rhs\n";
    cout << "ic = " << ic << ", sdiss = " << sdiss << ", cfric = " << cfric << "\n";
    cout << "grid = (" << theta_hat.get_grid().x << ", " << theta_hat.get_grid().y << "), block = (" << theta_hat.get_block().x << ", " << theta_hat.get_block().y << ")\n";
#endif //DEBUG
    //d_omega_ic_dummy<<<grid_my21_sec1, block_my21_sec1>>>(theta_y_hat.get_array_d(), strmf_hat.get_array_d(), omega_hat.get_array_d(0), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My / 2 + 1);
    d_omega_ic_sec1<<<grid_my21_sec1, block_my21_sec1>>>(theta_y_hat.get_array_d(0), strmf_hat.get_array_d(0), omega_hat.get_array_d(t_src), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My / 2 + 1);
#ifdef DEBUG
    gpuStatus();
#endif
}



void slab_cuda::dump_address()
{
    cout << "\nCompiled with NVCC\n";
    cout << "slab_cuda::dump_address()\n";
    cout << "\tconfig at " << (void*) &config << "\n";
    cout << "\tNx at " << (void*) &Nx << "\n";
    cout << "\tMy at " << (void*) &My << "\n";
    cout << "\ttlevs at " << (void*) &tlevs << "\n";
    cout << "\tplan_r2c at " << (void*) &plan_r2c << "\n";
    cout << "\tplan_c2r at " << (void*) &plan_c2r << "\n";
    //cout << "\tslab_output at " << (void*) &slab_output << "\n";
    cout << "\ttheta at " << (void*) &theta << "\n";
    cout << "\ttheta_x at " << (void*) &theta_x << "\n";
    cout << "\ttheta_y at " << (void*) &theta_y << "\n";
    //cout << "\tslab_output at " << (void*) &slab_output << "\n";
    cout << "\tstiff_params at " << (void*) &stiff_params << "\n";

}


// End of file slab_cuda.cu
