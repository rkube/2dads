/*
 * Only functions that require their own kernel
 *
 */

#include "include/slab_cuda.h"
#include <algorithm> //std::min, std::max


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
    //out[index].real = -1.0 * two_pi_L * double(row) * in[index].imag;
    //out[index].imag = two_pi_L * double(row) * in[index].real;
}


__global__
void d_d_dx_lo_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;
    if((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(1000000 + 1 * 1000 * col + row, row);
    out[index] = factor;
    //out[index].x = double(100000 + 1 * 1000 * col + row);
    //out[index].y = double(row);
}


// Frequencies: Nx/2
__global__
void d_d_dx_mid(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    //const int row = blockIdx.x * blockDim.x + threadIdx.x;
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

    cuda::cmplx_t factor(2000000 + 1000 * col + row);
    out[index] = factor;
    //out[index].x = double(200000 + 1000 * col + row);
    //out[index].y = 0.0;
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
    //out[index].x = -1.0 * two_pi_L * (double(row) - double(Nx)) * in[index].y;
    //out[index].y = two_pi_L * (double(row) - double(Nx)) * in[index].x;
}


__global__
void d_d_dx_up_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint index = row * My + col;
    if((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(3000000 + 1000 * col + double(row) - double(Nx), double(row) - double(Nx));
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
    //out[index].x = -1.0 * two_pi_L * double(col) * in[index].y;
    //out[index].y = two_pi_L * double(col) * in[index].x; 
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
    //out[index].x = 0.0;
    //out[index].y = 0.0;
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

    //double factor = -cuda::FOURPIS * (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2);
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;
    cuda::cmplx_t factor(-cuda::FOURPIS * (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2), 0.0);
    out[idx] = in[idx] / factor;
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;
}


__global__
void d_inv_laplace_sec1_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;
    
    cuda::cmplx_t factor(1000000 + 1000 * col + row, 1000 * col + row);
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
    //double factor = -cuda::FOURPIS * (
    //        ((double(row) - double(Nx)) * (double(row) - double(Nx))) * inv_Lx2 +
    //        (double(col * col) * inv_Ly2));
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;

    cuda::cmplx_t factor(-cuda::FOURPIS * (
            ((double(row) - double(Nx)) * (double(row) - double(Nx))) * inv_Lx2 +
            (double(col * col) * inv_Ly2)), 0.0);
    out[idx] = in[idx] / factor;
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;
}


__global__
void d_inv_laplace_sec2_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;

    cuda::cmplx_t factor(2000000.0 + 1000.0 * double(col) + double(row), 1000.0 * (double(row) - double(Nx)) + double(col));
    out[idx] = factor;
    //out[idx].x = double(200000 + 1000 * col + row);
    //out[idx].y = 1000.0 * (double(row) - double(Nx)) + double(col);
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

    //double factor = -cuda::FOURPIS * (
    //        (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2));
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;
    cuda::cmplx_t factor(-cuda::FOURPIS * (
            (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2)), 0.0);
    out[idx] = in[idx] / factor;
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;
}


__global__
void d_inv_laplace_sec3_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col; 
    if (row > Nx / 2 + 1)
        return;

    cuda::cmplx_t factor(3000000 + 1000 * col + row, 1000 * row + col);
    out[idx] = factor;
    //out[idx].x = double(300000 + 1000 * col + row);
    //out[idx].y = 1000. * double(row) + double(col);
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

    //double factor = -cuda::FOURPIS * (
    //        ((double(row) - double(Nx)) * (double(row) - double(Nx)) * inv_Lx2 + double(col * col) * inv_Ly2));
    //out[idx].x = in[idx].x / factor;
    //out[idx].y = in[idx].y / factor;
    cuda::cmplx_t factor(-cuda::FOURPIS * (
            ((double(row) - double(Nx)) * (double(row) - double(Nx)) * inv_Lx2 + double(col * col) * inv_Ly2)), 0.0);
    out[idx] = in[idx] / factor;
}


__global__
void d_inv_laplace_sec4_enumerate(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    //const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    //const uint idx = (row + 1) * My - 1;
    const uint col = My - 1;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = row * My + col; 

    if (row >= Nx)
        return;

    cuda::cmplx_t factor(double(4000000 + 1000 * col + row), 1000. * (double(row) - double(Nx)) + double(col));
    out[idx] = factor;
    //out[idx].x = double(400000 + 1000 * col + row);
    //out[idx].y = 1000. *(double(row) - double(Nx)) + double(col);
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

    cuda::cmplx_t value(1000 * col + row, 1);
    A[p.level - tlev][idx] = value;
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
    //cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    //cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    //cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));
    

    // Add contribution from explicit / implicit parts
    for(uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
        //sum1_alpha.x += A[p.level - k][idx].x * alpha[off_a - k];
        //sum1_alpha.y += A[p.level - k][idx].y * alpha[off_a - k];
        //sum1_beta.x += A_rhs[p.level - 1 - k][idx].x * beta[off_b - k];
        //sum1_beta.y += A_rhs[p.level - 1 - k][idx].y * beta[off_b - k];
    }
    //A[p.level - tlev][idx].x = (sum1_alpha.x + p.delta_t * sum1_beta.x) * temp_div; 
    //A[p.level - tlev][idx].y = (sum1_alpha.y + p.delta_t * sum1_beta.y) * temp_div; 
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

    cuda::cmplx_t value(1000 * col + row);
    A[p.level - tlev][idx] = value;
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
    //cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    //cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    //cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

    for(uint k = 1; k < tlev; k++)
    {
        //sum1_alpha.x += A[p.level - k][idx].x * alpha[off_a - k];
        //sum1_alpha.y += A[p.level - k][idx].y * alpha[off_a - k];
        //sum1_beta.x += A_rhs[p.level - 1 - k][idx].x * beta[off_b - k];
        //sum1_beta.y += A_rhs[p.level - 1 - k][idx].y * beta[off_b - k];
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    //A[p.level - tlev][idx].x = (sum1_alpha.x + p.delta_t * sum1_beta.x) * temp_div;
    //(A[p.level - tlev][idx].y = (sum1_alpha.y + p.delta_t * sum1_beta.y) * temp_div;
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

    cuda::cmplx_t value(1000 * col + row, 3);
    A[p.level - tlev][idx] = value;
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
    //cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    //cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

    for(uint k = 1; k < tlev; k++)
    {
        //sum1_alpha.x += A[p.level - k][idx].x * alpha[off_a - k];
        //sum1_alpha.y += A[p.level - k][idx].y * alpha[off_a - k];
        //sum1_beta.x += A_rhs[p.level - 1 - k][idx].x * beta[off_b - k];
        //sum1_beta.y += + A_rhs[p.level - 1 - k][idx].y * beta[off_b - k];
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    //A[p.level - tlev][idx].x = (sum1_alpha.x + p.delta_t * sum1_beta.x) * temp_div;
    //A[p.level - tlev][idx].y = (sum1_alpha.y + p.delta_t * sum1_beta.y) * temp_div;
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

    cuda::cmplx_t value(1000 * col + row, 4.0);
    A[p.level - tlev][idx] = value;
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
    //cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    //cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * (p.diff * (kx * kx + ky * ky) + p.hv * (kx*kx*kx*kx*kx*kx + ky*ky*ky*ky*ky*ky)));

    for(uint k = 1; k < tlev; k++)
    {
        //sum1_alpha.x += A[p.level - k][idx].x * alpha[off_a - k];
        //sum1_alpha.y += A[p.level - k][idx].y * alpha[off_a - k];
        //sum1_beta.x += A_rhs[p.level - 1 - k][idx].x * beta[off_b - k];
        //sum1_beta.y += A_rhs[p.level - 1 - k][idx].y * beta[off_b - k];
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    //A[p.level - tlev][idx].x = (sum1_alpha.x + p.delta_t * sum1_beta.x) * temp_div;
    //A[p.level - tlev][idx].y = (sum1_alpha.y + p.delta_t * sum1_beta.y) * temp_div;
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
}


__global__
void d_integrate_stiff_debug(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::real_t* alpha, cuda::real_t* beta, cuda::stiff_params_t p, uint tlev)
{
    //const uint col = 1;
    const uint row = 1;
    //const uint idx = row * p.My + col;
    const uint idx = 2;

    const cuda::real_t hv = 0.0001;
    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::TWOPI * cuda::real_t(row) / p.length_x;
    cuda::real_t ky = cuda::TWOPI * cuda::real_t(row) / p.length_y;
    printf("delta_t = %f, diff = %f\n", p.delta_t, p.diff);
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (alpha[(tlev - 2) * p.level] + p.delta_t * p.diff * (kx * kx + ky * ky) + hv * (kx*kx*kx*kx + ky*ky*ky*ky));

    printf("\ttlev = %d, off_a = %d, off_b = %d\n", tlev, off_a, off_b);
    for(uint k = 1; k < tlev; k++)
    {
        printf("\ttlev=%d,k=%d\t %f * A[%d] + dt * %f * A_R[%d]\n", tlev, k, alpha[off_a - k], p.level - k, beta[off_b - k], p.level - 1 - k);
        printf("\ttlev=%d, k = %d\t sum_alpha += %f * (%f, %f)\n", tlev, k, alpha[off_a - k], (A[p.level -k][idx]).re(), (A[p.level -k][idx]).im());
        printf("\ttlev=%d, k = %d\t sum_beta+= %f * (%f, %f)\n", tlev, k, beta[off_b - k], (A_rhs[p.level - 1 - k][idx]).re(), (A_rhs[p.level - 1 - k][idx]).im());
        //sum1_alpha.x += (A[p.level - k][idx]).x * alpha[off_a - k];
        //sum1_alpha.y += A[p.level - k][idx].y * alpha[off_a - k];
        //sum1_beta.x += A_rhs[p.level - 1 - k][idx].x * beta[off_b - k];
        //sum1_beta.y += A_rhs[p.level - 1 - k][idx].y * beta[off_b - k];
        sum_alpha += A[p.level - k][idx] * alpha[off_a - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * beta[off_b - k];
    }
    //sum1_beta = cuCmul(sum1_beta, make_cuDoubleComplex(p.delta_t, 0.0));
    //sum1_beta.x = sum1_beta.x * p.delta_t;
    //sum1_beta.y = sum1_beta.y * p.delta_t;
    //A[p.level - tlev][idx].x = (sum1_alpha.x + p.delta_t * sum1_beta.x) * temp_div;
    //A[p.level - tlev][idx].y = (sum1_alpha.y + p.delta_t * sum1_beta.y) * temp_div;
    //A[p.level - tlev][idx] = cuCmul(cuCadd(sum1_alpha, sum1_beta), temp_div);
    A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
    printf("\ttlev=%d, computing A[%d], gamma_0 = %f\n", tlev, p.level - tlev, alpha[(tlev - 2) * p.level]);
    printf("sum1_alpha = (%f, %f)\tsum1_beta = (%f, %f)\t", sum_alpha.re(), sum_alpha.im(), sum_beta.re(),  sum_beta.im());
    printf("temp_div = %f\n", temp_div); 
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev, idx, A[p.level - tlev][idx].re(), A[p.level - tlev][idx].im());
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
    //const uint col = 2;
    //const uint row = 0;
    //const uint idx = row * My + col;
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
}


// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint my21 = My / 2 + 1;
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

#ifdef DEBUG
    cout << "slab_chda::inv_laplace(...)\n";
    cout << "block_sec12 = (" << block_sec12.x << ", " << block_sec12.y << ")\t";
    cout << "grid_sec1 = (" << grid_sec1.x << ", " << grid_sec1.y << ")\n";
    cout << "grid_sec2 = (" << grid_sec2.x << ", " << grid_sec2.y << ")\n";
    cout << "grid_sec3 = (" << grid_sec3.x << ", " << grid_sec3.y << ")\n";
    cout << "grid_sec4 = (" << grid_sec4.x << ", " << grid_sec4.y << ")\n";
#endif //DEBUG

    d_inv_laplace_sec1<<<grid_sec1, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2<<<grid_sec2, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3<<<grid_sec3, block_sec3>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4<<<grid_sec4, block_sec4>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_zero<<<1, 1>>>(arr_out -> get_array_d(0));
    cudaDeviceSynchronize();
}

// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace_enumerate(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_src)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t, cuda::real_t>* arr_out = get_field_by_name(dst_name);

    const uint my21 = My / 2 + 1;
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

#ifdef DEBUG
    cout << "slab_chda::inv_laplace(...)\n";
    cout << "block_sec12 = (" << block_sec12.x << ", " << block_sec12.y << ")\t";
    cout << "grid_sec1 = (" << grid_sec1.x << ", " << grid_sec1.y << ")\n";
    cout << "grid_sec2 = (" << grid_sec2.x << ", " << grid_sec2.y << ")\n";
    cout << "grid_sec3 = (" << grid_sec3.x << ", " << grid_sec3.y << ")\n";
    cout << "grid_sec4 = (" << grid_sec4.x << ", " << grid_sec4.y << ")\n";
#endif //DEBUG

    d_inv_laplace_sec1_enumerate<<<grid_sec1, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2_enumerate<<<grid_sec2, block_sec12>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3_enumerate<<<grid_sec3, block_sec3>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4_enumerate<<<grid_sec4, block_sec4>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), Nx, my21, inv_Lx2, inv_Ly2);
    d_inv_laplace_zero<<<1, 1>>>(arr_out -> get_array_d(0));
    cudaDeviceSynchronize();
}


void slab_cuda :: integrate_stiff(twodads::dyn_field_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname); 
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname); 
    //d_integrate_stiff_debug<<<1, 1>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec1<<<grid_sec1, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2<<<grid_sec2, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3<<<grid_sec3, block_sec3>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4<<<grid_sec4, block_sec4>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    cudaDeviceSynchronize();
}


void slab_cuda :: integrate_stiff_enumerate(twodads::dyn_field_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t, cuda::real_t>* A = get_field_by_name(fname);
    cuda_array<cuda::cmplx_t, cuda::real_t>* A_rhs = get_rhs_by_name(fname);
    cout << "integrate_stiff_enumerate\n";
    d_integrate_stiff_sec1_enumerate<<<grid_sec1, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2_enumerate<<<grid_sec2, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3_enumerate<<<grid_sec3, block_sec3>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4_enumerate<<<grid_sec4, block_sec4>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    cudaDeviceSynchronize();
} 


// Compute RHS for Navier stokes model, store result in time index 0 of theta_rhs_hat
void slab_cuda :: theta_rhs_ns(uint t_src)
{
    //cout << "theta_rhs_ns\n";
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


void slab_cuda :: theta_rhs_lin(uint t_src)
{
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


///@brief Compute explicit part for Hasegawa-Wakatani model, store result in time index 0 of theta_rhs_hat
///@detailed $\mathcal{N}^0 = \left{n, \phi\right} + \mathcal{C} (\widetilde{phi} - n) - \phi_y$
void slab_cuda :: theta_rhs_hw(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    // Poisson bracket is on the RHS: dn/dt = {n, phi} + ...
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    cudaDeviceSynchronize();
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
    //cout << "theta_rhs_hw: theta_rhs_Hat = \n" << theta_rhs_hat << "\n";
    //d_theta_rhs_hw_debug<<<1, 1>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), strmf_y_hat.get_array_d(), C, Nx, My / 2 + 1);
    d_theta_rhs_hw<<<grid_my21_sec1, block_my21_sec1>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), strmf_y_hat.get_array_d(), C, Nx, My / 2 + 1);
    cudaDeviceSynchronize();
}


/// @brief Explicit part for the MHW model
/// @detailed $\mathcal{N}^t = \left{n, \phi\right} - \mathcal{C} (\widetilde{phi} - \widetilde{n}) - \phi_y$
void slab_cuda :: theta_rhs_hwmod(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    cout << "theta_rhs_mod, C = " << C << "\n";
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    cudaDeviceSynchronize();
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
    // Neglect ky=0 modes for in coupling term
    d_coupling_hwmod<<<grid_my21_sec1, block_my21_sec1>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, Nx, My / 2 + 1);
    cudaDeviceSynchronize();
    theta_rhs_hat -= strmf_y_hat; 
    cudaDeviceSynchronize();
}

void slab_cuda :: theta_rhs_log(uint t_src)
{
    d_pbracket<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    d_theta_rhs_log<<<grid_nx_my, block_nx_my>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


void slab_cuda :: omega_rhs_ns(uint t_src)
{
    d_pbracket<<<grid_nx_my, block_nx_my>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    //cout << "omega_rhs_ns\n";
}

/// @brief RHS for the Hasegawa-Wakatani model
/// @detailed RHS = {Omega, phi} - C(phi - n)
void slab_cuda :: omega_rhs_hw(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    cout << "omega_rhs_hw, C = " << C << "\n";
    d_pbracket<<<grid_nx_my, block_nx_my>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    cudaDeviceSynchronize();
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    cudaDeviceSynchronize();
    d_omega_rhs_hw<<<grid_my21_sec1, block_my21_sec1>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, Nx, My / 2 + 1);
    cudaDeviceSynchronize();
}


void slab_cuda :: omega_rhs_hwmod(uint t_src)
{
    cuda::real_t C = config.get_model_params(2);
    cout << "omega_rhs_hwmod, C = " << C << "\n";
    d_pbracket<<<grid_nx_my, block_nx_my>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    cudaDeviceSynchronize();
    dft_r2c(twodads::f_tmp, twodads::f_omega_rhs_hat, 0);
    cudaDeviceSynchronize();
    d_coupling_hwmod<<<grid_my21_sec1, block_my21_sec1>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, Nx, My / 2 + 1);
    cudaDeviceSynchronize();
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
    cout << "ic = (" << ic.x << ", " << ic.y << "), sdiss = (" << sdiss.x << ", " << sdiss.y << "), cfric = (" << cfric.x << ", " << cfric.y << ")\n";
    cout << "grid = (" << theta_hat.get_grid().x << ", " << theta_hat.get_grid().y << "), block = (" << theta_hat.get_block().x << ", " << theta_hat.get_block().y << ")\n";
#endif //DEBUG
    //d_omega_ic_dummy<<<grid_my21_sec1, block_my21_sec1>>>(theta_y_hat.get_array_d(), strmf_hat.get_array_d(), omega_hat.get_array_d(0), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My / 2 + 1);
    d_omega_ic_sec1<<<grid_my21_sec1, block_my21_sec1>>>(theta_y_hat.get_array_d(0), strmf_hat.get_array_d(0), omega_hat.get_array_d(t_src), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My / 2 + 1);
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
    cout << "\tslab_output at " << (void*) &slab_output << "\n";
    cout << "\ttheta at " << (void*) &theta << "\n";
    cout << "\ttheta_x at " << (void*) &theta_x << "\n";
    cout << "\ttheta_y at " << (void*) &theta_y << "\n";
    cout << "\tslab_output at " << (void*) &slab_output << "\n";
    cout << "\tstiff_params at " << (void*) &stiff_params << "\n";

}


// End of file slab_cuda2.cu
