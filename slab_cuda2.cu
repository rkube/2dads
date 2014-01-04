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

// d/dx: Frequencies 0 .. N/2 - 1
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
    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(row)));
}


// Frequencies: Nx/2
__global__
void d_d_dx_mid(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Lx)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    //const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = Nx / 2;
    const uint index = row * My + col;

    //const double two_pi_L = 2.0 * cuda::PI / Lx;
    // Return if we don't have an item to work on
    if((col >= My) || (row >= Nx))
        return;
    out[index] = make_cuDoubleComplex(0.0, 0.0);
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
    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * (double(row) - double(Nx))));
}


// Frequencies 0..My / 2
__global__
void d_d_dy_lo(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Ly)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * My + col;

    if ((col >= My) || (row >= Nx))
        return;
    double two_pi_L = cuda::TWOPI / Ly;

    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, two_pi_L * double(col)));
}


__global__
void d_d_dy_up(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double Ly)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= Nx)
        return;

    index = (index + 1) * My - 1;
    out[index] = cuCmul(in[index], make_cuDoubleComplex(0.0, 0.0));
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


__global__
//void d_inv_laplace_sec1(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2, double four_pi_s)
void d_inv_laplace_sec1(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;

    double factor = -cuda::FOURPIS * (double(row * row) * inv_Lx2 + double(col * col) * inv_Ly2);
    out[idx] = cuCdiv(in[idx], make_cuDoubleComplex(factor, 0.0));
    //out[idx] = make_cuDoubleComplex(double(row), double(col));
}


__global__
void d_inv_laplace_sec2(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
        return;
    double factor = -cuda::FOURPIS * (
            ((double(row) - double(Nx)) * (double(row) - double(Nx))) * inv_Lx2 +
            (double(col * col) * inv_Ly2));
    out[idx] = cuCdiv(in[idx], make_cuDoubleComplex(factor, 0.0));
    //out[idx] = make_cuDoubleComplex(double(row) - double(Nx), double(col));
}


__global__
void d_inv_laplace_sec3(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2)
{
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = (row + 1) * My - 1;

    if (row > Nx / 2 + 1)
        return;

    double factor = -cuda::FOURPIS * (
            (double(row * row) * inv_Lx2 + double(My * My) * inv_Ly2));
    out[idx] = cuCdiv(in[idx], make_cuDoubleComplex(factor, 0.0));
    //out[idx] = make_cuDoubleComplex(double(row), double(My));
}


__global__
void d_inv_laplace_sec4(cuda::cmplx_t* in, cuda::cmplx_t* out, uint Nx, uint My, double inv_Lx2, double inv_Ly2) 
{
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + Nx / 2 + 1;
    const uint idx = (row + 1) * My - 1;

    if (row >= Nx)
        return;

    double factor = -cuda::FOURPIS * (
            ((double(row) - double(Nx)) * (double(row) - double(Nx)) * inv_Lx2 + double(My * My) * inv_Ly2));
    out[idx] = cuCdiv(in[idx], make_cuDoubleComplex(factor, 0.0));
    //out[idx] = make_cuDoubleComplex(double(row) - double(Nx), double(My));
}


__global__
void d_inv_laplace_zero(cuda::cmplx_t* out)
{
    out[0] = make_cuDoubleComplex(0.0, 0.0);
}


/*
 * Stiffly stable time integration
 * temp = sum(k=1..level) alpha[T-2][level-k] * u^{T-k} + delta_t * beta[T-2][level -k - 1] * u_RHS^{T-k-1}
 * u^{n+1}_{i} = temp / (alpha[T-2][0] + delta_t * diff * (kx^2 + ky^2))
 *
 * Use same sector splitting as for inv_laplace
 */


__global__
void d_integrate_stiff_sec1(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::cmplx_t* alpha, cuda::cmplx_t* beta, cuda::stiff_params_t p, uint tlev) 
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
    cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t temp_div = make_cuDoubleComplex(1. / (alpha[(tlev - 2) * p.level].x + p.delta_t * p.diff * (kx * kx + ky * ky)), 0.0);

    // Add contribution from explicit / implicit parts
    for(uint k = 1; k < tlev; k++)
    {
        sum1_alpha = cuCadd(sum1_alpha, cuCmul(A[p.level - k][idx], alpha[off_a - k]));
        sum1_beta = cuCadd(sum1_beta, cuCmul(A_rhs[p.level - 1 - k][idx], beta[off_b - k]));
    }
    sum1_beta = cuCmul(sum1_beta, make_cuDoubleComplex(p.delta_t, 0.0));
    A[p.level - tlev][idx] = cuCmul(cuCadd(sum1_alpha, sum1_beta), temp_div);
}


__global__
void d_integrate_stiff_sec2(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::cmplx_t* alpha, cuda::cmplx_t* beta, cuda::stiff_params_t p, uint tlev)
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
    cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t temp_div = make_cuDoubleComplex(1. / (alpha[(tlev - 2) * p.level].x + p.delta_t * p.diff * (kx * kx + ky * ky)), 0.0);

    for(uint k = 1; k < tlev; k++)
    {
        sum1_alpha = cuCadd(sum1_alpha, cuCmul(A[p.level - k][idx], alpha[off_a - k]));
        sum1_beta = cuCadd(sum1_beta, cuCmul(A_rhs[p.level - 1 - k][idx], beta[off_b - k]));
    }
    sum1_beta = cuCmul(sum1_beta, make_cuDoubleComplex(p.delta_t, 0.0));
    A[p.level - tlev][idx] = cuCmul(cuCadd(sum1_alpha, sum1_beta), temp_div);
}


__global__
void d_integrate_stiff_sec3(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::cmplx_t* alpha, cuda::cmplx_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = p.My;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = (row + 1) * p.My - 1; 
    if (row >= p.Nx)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(row) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(col) * cuda::TWOPI/ p.length_y;
    cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t temp_div = make_cuDoubleComplex(1. / (alpha[(tlev - 2) * p.level].x + p.delta_t * p.diff * (kx * kx + ky * ky)), 0.0);

    for(uint k = 1; k < tlev; k++)
    {
        sum1_alpha = cuCadd(sum1_alpha, cuCmul(A[p.level - k][idx], alpha[off_a - k]));
        sum1_beta = cuCadd(sum1_beta, cuCmul(A_rhs[p.level - 1 - k][idx], beta[off_b - k]));
    }
    sum1_beta = cuCmul(sum1_beta, make_cuDoubleComplex(p.delta_t, 0.0));
    A[p.level - tlev][idx] = cuCmul(cuCadd(sum1_alpha, sum1_beta), temp_div);
}


__global__
void d_integrate_stiff_sec4(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::cmplx_t* alpha, cuda::cmplx_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = p.My;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x + p.Nx / 2 + 1;
    const uint idx = (row + 1) * p.My - 1; 
    if (row >= p.Nx)
        return;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = (cuda::real_t(row) - cuda::real_t(p.Nx)) * cuda::TWOPI / p.length_x;
    cuda::real_t ky = cuda::real_t(col) * cuda::TWOPI / p.length_y;
    cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t temp_div = make_cuDoubleComplex(1. / (alpha[(tlev - 2) * p.level].x + p.delta_t * p.diff * (kx * kx + ky * ky)), 0.0);

    for(uint k = 1; k < tlev; k++)
    {
        sum1_alpha = cuCadd(sum1_alpha, cuCmul(A[p.level - k][idx], alpha[off_a - k]));
        sum1_beta = cuCadd(sum1_beta, cuCmul(A_rhs[p.level - 1 - k][idx], beta[off_b - k]));
    }
    sum1_beta = cuCmul(sum1_beta, make_cuDoubleComplex(p.delta_t, 0.0));
    A[p.level - tlev][idx] = cuCmul(cuCadd(sum1_alpha, sum1_beta), temp_div);
}


__global__
void d_integrate_stiff_dummy(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::cmplx_t* alpha, cuda::cmplx_t* beta, cuda::stiff_params_t p, uint tlev)
{
    const uint col = 1;
    const uint row = 1;
    const uint idx = row * p.My + col;

    uint off_a = (tlev - 2) * p.level + tlev;
    uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::TWOPI * cuda::real_t(row) / p.length_x;
    cuda::real_t ky = cuda::TWOPI * cuda::real_t(row) / p.length_y;
    printf("delta_t = %f, diff = %f\n", p.delta_t, p.diff);
    cuda::cmplx_t sum1_alpha = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t sum1_beta = make_cuDoubleComplex(0.0, 0.0);
    cuda::cmplx_t temp_div = make_cuDoubleComplex(1. / (alpha[(tlev - 2) * p.level].x + p.delta_t * p.diff * (kx * kx + ky * ky)), 0.0);

    for(uint k = 1; k < tlev; k++)
    {
        printf("tlev=%d,k=%d\t %f * A[%d] + dt * %f * A_R[%d]\n", tlev, k, (alpha[off_a - k]).x, p.level - k, (beta[off_b - k]).x, p.level - 1 - k);
        printf("tlev=%d, k = %d\t sum_alpha += %f * (%f, %f)\n", tlev, k, (alpha[off_a - k]).x, (A[p.level -k][idx]).x, (A[p.level -k][idx]).y);
        printf("tlev=%d, k = %d\t sum_beta+= %f * (%f, %f)\n", tlev, k, (beta[off_b - k]).x, (A_rhs[p.level - 1 - k][idx]).x, (A_rhs[p.level - 1 - k][idx]).y);
        sum1_alpha = cuCadd(sum1_alpha, cuCmul(A[p.level - k][idx], alpha[off_a - k]));
        sum1_beta = cuCadd(sum1_beta, cuCmul(A_rhs[p.level - 1 - k][idx], beta[off_b - k]));
    }
    sum1_beta = cuCmul(sum1_beta, make_cuDoubleComplex(p.delta_t, 0.0));
    A[p.level - tlev][idx] = cuCmul(cuCadd(sum1_alpha, sum1_beta), temp_div);
    printf("\ttlev=%d, computing A[%d], gamma_0 = %f\n", tlev, p.level - tlev, alpha[(tlev - 2) * p.level].x);
    printf("sum1_alpha = (%f, %f)\tsum1_beta = (%f, %f)\t", sum1_alpha.x, sum1_alpha.y, sum1_beta.x, sum1_beta.y);
    printf("temp_div = (%f, %f)\n", temp_div.x, temp_div.y);
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev, idx, A[p.level - tlev][idx].x, A[p.level - tlev][idx].y);
}


/*
 *
 * Kernels to compute non-linear operators
 *
 */


// Poisson brackt: theta_x * strmf_y - theta_x * strmf_y
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

// RHS for vorticity eq, interchange turbulence
// RHS = RHS - int * theta_y - sdiss * strmf - collfric * omega
__global__
void d_omega_ic(cuda::cmplx_t* theta_y_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* omega_hat, cuda::cmplx_t ic, cuda::cmplx_t sdiss, cuda::cmplx_t cfric, cuda::cmplx_t* out, uint Nx, uint My)
{
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My + col;
    if ((col >= My) || (row >= Nx))
       return;

    cuda::cmplx_t part1 = cuCmul(ic, theta_y_hat[idx]);
    cuda::cmplx_t part2 = cuCmul(sdiss, strmf_hat[idx]);
    cuda::cmplx_t part3 = cuCmul(cfric, omega_hat[idx]);
    out[idx] = cuCsub(out[idx], part1);
    out[idx] = cuCsub(out[idx], part2);
    out[idx] = cuCsub(out[idx], part3);
    //out[idx] = out[idx] - ic * theta_y[idx] - sdiss * strmf[idx] - cfric * omega[idx];

}


__global__
void d_omega_ic_dummy(cuda::cmplx_t* theta_y_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* omega_hat, cuda::cmplx_t ic, cuda::cmplx_t sdiss, cuda::cmplx_t cfric, cuda::cmplx_t* out, uint Nx, uint My)
{
    const uint col = 0;
    const uint row = 0;
    const uint idx = 1;
    if((col >= My) || (row >= Nx))
        return;
    printf("d_omega_ic_dummy\n");
    printf("theta_y_hat = (%f, %f), strmf_hat = (%f, %f), omega_hat = (%f, %f)\n", (theta_y_hat[idx]).x, (theta_y_hat[idx]).y, (strmf_hat[idx]).x, (strmf_hat[idx]).y, (omega_hat[idx]).x, (omega_hat[idx]).x);
    printf("ic = (%f, %f), sdiss = (%f, %f), cfric = (%f, %f)\n", ic.x, ic.y, sdiss.x, sdiss.y, cfric.x, cfric.y);
    printf("omega_rhs_hat = (%f, %f)\n", (out[idx]).x, (out[idx]).y);
    cuda::cmplx_t part1 = cuCmul(ic, theta_y_hat[idx]);
    out[idx] = cuCsub(out[idx], part1);
    printf("part1 = (%f, %f), out[idx] = (%f, %f)", part1.x, part1.y, out[idx].x, out[idx].y);

    cuda::cmplx_t part2 = cuCmul(sdiss, strmf_hat[idx]);
    out[idx] = cuCsub(out[idx], part2);
    
    cuda::cmplx_t part3 = cuCmul(cfric, omega_hat[idx]);
    out[idx] = cuCsub(out[idx], part3);

    printf("omega_rhs_hat = (%f, %f)\n", (out[idx]).x, (out[idx]).y);
}

/*****************************************************************************
 *
 * Function implementation
 *
 ****************************************************************************/


void slab_cuda :: d_dx(twodads::field_k_t src_name, twodads::field_k_t dst_name)
{
    cuda_array<cuda::cmplx_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t>* arr_out = get_field_by_name(dst_name);
    
    const uint Nx = config.get_nx();
    const uint My = config.get_my();
    double Lx = config.get_deltax() * double(Nx);
    dim3 grid_dx_half(Nx / 2, arr_in -> get_grid().y);
    dim3 grid_dx_single(1, arr_in -> get_grid().y);

    //cout << "Lx = " << Lx << "\ngrid_dx_half = (" << grid_dx_half.x << ", " << grid_dx_half.y << ")\t";
    //cout << "grid_dx_single = (" << grid_dx_single.x << ", " << grid_dx_single.y << ")\n";
    
    d_d_dx_lo<<<grid_dx_half, arr_in -> get_block()>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Lx);
    d_d_dx_mid<<<grid_dx_single, arr_in -> get_block()>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Lx);
    d_d_dx_up<<<grid_dx_half, arr_in -> get_block()>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Lx);
    cudaDeviceSynchronize();    
}


void slab_cuda :: d_dy(twodads::field_k_t src_name, twodads::field_k_t dst_name)
{
    cuda_array<cuda::cmplx_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t>* arr_out = get_field_by_name(dst_name);

    const uint Nx = config.get_nx();
    const uint My = config.get_my();
    double Ly = config.get_lengthy();
    //dim3 block_single(cuda::cuda_blockdim_nx);
    //dim3 grid_single(Nx / cuda::cuda_blockdim_nx);

    d_d_dy_lo<<<arr_in -> get_grid(), arr_out -> get_block()>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Ly);
    //d_d_dy_up<<<grid_single, block_single>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Ly);
    d_d_dy_up<<<grid_my21_sec2, block_my21_sec2>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Ly);
    cudaDeviceSynchronize();    
}


void slab_cuda :: inv_laplace(twodads::field_k_t src_name, twodads::field_k_t dst_name, uint t_in)
{
    cuda_array<cuda::cmplx_t>* arr_in = get_field_by_name(src_name);
    cuda_array<cuda::cmplx_t>* arr_out = get_field_by_name(dst_name);

    const uint Nx = config.get_nx();
    const uint My = config.get_my();
    const double inv_Lx2 = 1. / (config.get_lengthx() * config.get_lengthx());
    const double inv_Ly2 = 1. / (config.get_lengthy() * config.get_lengthy());

    //cout << "slab_chda::inv_laplace(...)\n";
    //cout << "block_sec12 = (" << block_sec12.x << ", " << block_sec12.y << ")\t";
    //cout << "grid_sec1 = (" << grid_sec1.x << ", " << grid_sec1.y << ")\n";
    //cout << "grid_sec2 = (" << grid_sec2.x << ", " << grid_sec2.y << ")\n";
    //cout << "grid_sec3 = (" << grid_sec3.x << ", " << grid_sec3.y << ")\n";
    //cout << "grid_sec4 = (" << grid_sec4.x << ", " << grid_sec4.y << ")\n";

    d_inv_laplace_sec1<<<grid_sec1, block_sec12>>>(arr_in -> get_array_d(t_in), arr_out -> get_array_d(0), Nx, My / 2 + 1, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec2<<<grid_sec2, block_sec12>>>(arr_in -> get_array_d(t_in), arr_out -> get_array_d(0), Nx, My / 2 + 1, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec3<<<grid_sec3, block_sec3>>>(arr_in -> get_array_d(t_in), arr_out -> get_array_d(0), Nx, My / 2 + 1, inv_Lx2, inv_Ly2);
    d_inv_laplace_sec4<<<grid_sec4, block_sec4>>>(arr_in -> get_array_d(t_in), arr_out -> get_array_d(0), Nx, My / 2 + 1, inv_Lx2, inv_Ly2);
    d_inv_laplace_zero<<<1, 1>>>(arr_out -> get_array_d(0));
    cudaDeviceSynchronize();    
}


void slab_cuda :: integrate_stiff(twodads::dyn_field_t fname, uint tlev)
{
    cuda_array<cuda::cmplx_t>* A = get_field_by_name(fname); 
    cuda_array<cuda::cmplx_t>* A_rhs = get_rhs_by_name(fname); 

    //d_integrate_stiff_dummy<<<1, 1>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec1<<<grid_sec1, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec2<<<grid_sec2, block_sec12>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec3<<<grid_sec3, block_sec3>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    d_integrate_stiff_sec4<<<grid_sec4, block_sec4>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev);
    cudaDeviceSynchronize();
}

void slab_cuda :: theta_rhs_lin()
{
    cout << "theta_rhs_lin\n";
    d_pbracket<<<theta.get_grid(), theta.get_block()>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}

void slab_cuda :: theta_rhs_log()
{
    cout << "theta_rhs_log\n";
    d_theta_rhs_log<<<theta.get_grid(), theta.get_block()>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_theta_rhs_hat, 0);
}


void slab_cuda::omega_rhs_ic()
{
    cout << "omega_rhs\n";
    // Compute Poisson bracket in real space, use full grid/block
    d_pbracket<<<theta.get_grid(), theta.get_block()>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), Nx, My);
    dft_r2c(twodads::f_tmp, twodads::f_tmp_hat, 0);
    // Convert model parameters to complex numbers
    cuda::cmplx_t ic = make_cuDoubleComplex(config.get_model_param(2), 0.0);
    cuda::cmplx_t sdiss = make_cuDoubleComplex(config.get_model_param(3), 0.0);
    cuda::cmplx_t cfric = make_cuDoubleComplex(config.get_model_param(4), 0.0);
    cout << "ic = (" << ic.x << ", " << ic.y << "), sdiss = (" << sdiss.x << ", " << sdiss.y << "), cfric = (" << cfric.x << ", " << cfric.y << ")\n";
    cout << "grid = (" << theta_hat.get_grid().x << ", " << theta_hat.get_grid().y << "), block = (" << theta_hat.get_block().x << ", " << theta_hat.get_block().y << ")\n";
    //d_omega_ic_dummy<<<1, 1>>>(theta_y_hat.get_array_d(), strmf_hat.get_array_d(), omega_hat.get_array_d(0), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My);
    d_omega_ic<<<theta_hat.get_grid(), theta_hat.get_block()>>>(theta_y_hat.get_array_d(0), strmf_hat.get_array_d(0), omega_hat.get_array_d(0), ic, sdiss, cfric, omega_rhs_hat.get_array_d(0), Nx, My / 2 + 1);
    //d_d_dy_lo<<<arr_in -> get_grid(), arr_out -> get_block()>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Ly);
    //d_d_dy_up<<<grid_single, block_single>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Ly);
    //d_d_dy_up<<<grid_my21_sec2, block_my21_sec2>>>(arr_in -> get_array_d(0), arr_out -> get_array_d(0), Nx, My / 2 + 1, Ly);
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
