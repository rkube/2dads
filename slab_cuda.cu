#include "error.h"
#include "slab_cuda.h"

using namespace std;
/*****************************************************************************
 *
 * Kernel implementation
 *
 ****************************************************************************/

// ky=0 modes are stored in the first row, Nx21 columns
__global__
void d_kill_ky0(cuda::cmplx_t* in, const uint My, const uint Nx21)
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < Nx21)    
        in[col] = cuda::cmplx_t(0.0, 0.0);
}

/*
 * Stiffly stable time integration
 * temp = sum(k=1..level) alpha[T-2][level-k] * u^{T-k} + delta_t * beta[T-2][level -k - 1] * u_RHS^{T-k-1}
 * u^{n+1}_{i} = temp / (alpha[T-2][0] + delta_t * diff * (kx^2 + ky^2))
 *
 * Use wave-numbers for even-ordered derivatives, i.e. don't round down to zero on
 * wavenumber My/2, Nx/2, as for first derivatives
 */


__global__
void d_integrate_stiff_map_2(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::stiff_params_t p)
{
	const int row = d_get_row();
	const int col = d_get_col();
	const int idx = row * p.Nx21 + col;

	cuda::real_t kx = cuda::TWOPI * cuda::real_t(col) / p.length_x;
	cuda::real_t ky = cuda::TWOPI * ((row < p.My / 2 + 1) ? cuda::real_t(row) : (cuda::real_t(row) - cuda::real_t(p.My))) / p.length_y;
	cuda::real_t k2 = kx * kx + ky * ky;
	cuda::real_t temp_div{0.0};

	cuda::cmplx_t dummy(0.0, 0.0);
	if((row < p.My) && (col < p.Nx21))
	{
		temp_div = 1.0 / (1.0 + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));


		dummy = A[3][idx];
		A[2][idx] = (A[3][idx] + A_rhs[2][idx] * p.delta_t) * temp_div;
		//A[2][idx] = (A[3][idx] + (A_rhs[2][idx] * cuda::ss3_beta_d[0][0] * p.delta_t)) * temp_div;
//		printf("\tcol = %d, threadIdx.x = %d, row = %d, blockIdx.y = %d, idx = %d, kx = %f, ky = %f, temp_div = %f\tupdating (%f, %f)->  (%f, %f)\n",
//				col, threadIdx.x, row, blockIdx.y, idx, kx, ky, temp_div,
//				dummy.re(), dummy.im(), A[2][idx].re(), A[2][idx].im());

	}
	return;
}


__global__
void d_integrate_stiff_map_3(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::stiff_params_t p)
{
	const int row = d_get_row();
	const int col = d_get_col();
	const int idx = row * p.Nx21 + col;

	cuda::real_t kx = cuda::TWOPI * cuda::real_t(col) / p.length_x;
	cuda::real_t ky = cuda::TWOPI * ((row < p.My/ 2 + 1) ? cuda::real_t(row) : (cuda::real_t(row) - cuda::real_t(p.My))) / p.length_y;
	cuda::real_t k2 = kx * kx + ky * ky;
	cuda::real_t temp_div{0.0};
	cuda::cmplx_t sum_alpha(0.0, 0.0);
	cuda::cmplx_t sum_beta(0.0, 0.0);

	if((row < p.My) && (col < p.Nx21))
	{
		temp_div = 1. / (1.5 + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

		//sum_alpha = (A[3][idx] * cuda::ss3_alpha_d[1][2]) + (A[2][idx] * cuda::ss3_alpha_d[1][1]);
		//sum_beta = (A_rhs[2][idx] * cuda::ss3_beta_d[1][1]) + (A_rhs[1][idx] * cuda::ss3_beta_d[1][0]);
		sum_alpha = A[3][idx] * (-0.5) + A[2][idx] * 2.0; 
		sum_beta = A_rhs[2][idx] * (-1.0) + A_rhs[1][idx] * (2.0);
		A[1][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
	}
	return;
}

__global__
void d_integrate_stiff_map_4(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::stiff_params_t p)
{
	const int row = d_get_row();
	const int col = d_get_col();
	const int idx = row * p.Nx21 + col;

	cuda::real_t kx = cuda::TWOPI * cuda::real_t(col) / p.length_x;
	cuda::real_t ky = cuda::TWOPI * ((row < p.My/ 2 + 1) ? cuda::real_t(row) : (cuda::real_t(row) - cuda::real_t(p.My))) / p.length_y;
	cuda::real_t k2 = kx * kx + ky * ky;
	cuda::real_t temp_div{0.0};
	cuda::cmplx_t sum_alpha(0.0, 0.0);
	cuda::cmplx_t sum_beta(0.0, 0.0);

	if((row < p.My) && (col < p.Nx21))
	{
		temp_div = 1. / ((11. / 6.) + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

		sum_alpha = A[3][idx] * (1. / 3.) - A[2][idx] * 1.5 + A[1][idx] * 3.;
		sum_beta = A_rhs[2][idx] - A_rhs[1][idx] * 3. + A_rhs[0][idx] * 3.; 
		A[0][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;

	}
	return;
}


__global__
void d_integrate_stiff_map_4_debug(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::stiff_params_t p)
{
	const int row = 2;
	const int col = 0;
	const int idx = row * p.Nx21 + col;

	const cuda::real_t kx = cuda::TWOPI * cuda::real_t(col) / p.length_x;
	const cuda::real_t ky = cuda::TWOPI * ((row < p.My/ 2 + 1) ? cuda::real_t(row) : (cuda::real_t(p.My) - cuda::real_t(row))) / p.length_y;
	const cuda::real_t k2 = kx * kx + ky * ky;
	cuda::real_t temp_div{0.0};
	cuda::cmplx_t sum_alpha(0.0, 0.0);
	cuda::cmplx_t sum_beta(0.0, 0.0);
	cuda::cmplx_t dummy(0.0, 0.0);

	temp_div = 1. / (cuda::ss3_alpha_d[2][0] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

	sum_alpha = A[3][idx] * cuda::ss3_alpha_d[2][2] + A[2][idx] * cuda::ss3_alpha_d[2][1] + A[1][idx] * cuda::ss3_alpha_d[2][0];
	sum_beta = A_rhs[2][idx] * cuda::ss3_beta_d[2][2] + A_rhs[1][idx] * cuda::ss3_beta_d[2][1] + A_rhs[0][idx] * cuda::ss3_beta_d[2][0];

	printf("\td_integrate_stiff_map_4_debug\n");
	printf("\tsum_alpha = (%f, %f) * %f\n", A[3][idx].re(), A[3][idx].im(), cuda::ss3_alpha_d[2][2]);
	printf("\t          + (%f, %f) * %f\n", A[2][idx].re(), A[2][idx].im(), cuda::ss3_alpha_d[2][1]);
	printf("\t          + (%f, %f) * %f\n", A[1][idx].re(), A[1][idx].im(), cuda::ss3_alpha_d[2][0]);
	printf("\t          = (%f, %f)\n\n", sum_alpha.re(), sum_alpha.im());
	printf("\tsum_beta = (%f, %f) * %f\n", A_rhs[2][idx].re(), A_rhs[2][idx].im(), cuda::ss3_beta_d[2][2]);
	printf("\t         + (%f, %f) * %f\n", A_rhs[1][idx].re(), A_rhs[1][idx].im(), cuda::ss3_beta_d[2][1]);
	printf("\t         + (%f, %f) * %f\n", A_rhs[0][idx].re(), A_rhs[0][idx].im(), cuda::ss3_beta_d[2][0]);
	printf("\t         = (%f, %f)\n\n", sum_beta.re(), sum_beta.im());

	dummy = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;

	printf("\tA[0][%d] = (%f, %f) + ((%f, %f) * %f) * %f\n", idx,
														   sum_alpha.re(), sum_alpha.im(),
														   sum_beta.re(), sum_beta.im(),
														   p.delta_t, temp_div);
	printf("\t         = (%f, %f)\n", dummy.re(), dummy.im());
}


__global__
void d_integrate_stiff_ky0(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::stiff_params_t p, uint tlev)
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = col;


    //uint off_a = (tlev - 2) * p.level + tlev;
    //uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::real_t(col) * cuda::TWOPI / p.length_x;
    // ky = 0
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    //cuda::real_t temp_div =  1. / (cuda::ss3_alpha_d[(tlev - 2) * p.level] + p.delta_t * (p.diff * kx * kx + p.hv * kx * kx * kx * kx * kx * kx));
    cuda::real_t temp_div =  1. / (cuda::ss3_alpha_d[tlev - 2][0] + p.delta_t * (p.diff * kx * kx + p.hv * kx * kx * kx * kx * kx * kx));

    for (uint k = 1; k < tlev; k++)
    {
        sum_alpha += A[p.level - k][idx] * cuda::ss3_alpha_d[tlev - 2][tlev - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * cuda::ss3_beta_d[tlev - 2][tlev - k - 1];
    }
    if (col < p.Nx21)
    {
    	A[p.level - tlev][idx] = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
    }
}


// Print very verbose debug information of what stiffk does
// Do no update A!
__global__
void d_integrate_stiff_debug(cuda::cmplx_t** A, cuda::cmplx_t** A_rhs, cuda::stiff_params_t p, uint tlev, uint row, uint col)
{
    col = col % p.Nx21;
    row = row % p.My;


    const uint idx = row * p.Nx21 + col;
    //uint off_a = (tlev - 2) * p.level + tlev;
    //uint off_b = (tlev - 2) * (p.level - 1) + tlev - 1;
    cuda::real_t kx = cuda::TWOPI * cuda::real_t(col) / p.length_x;
    cuda::real_t ky = cuda::TWOPI * (row < p.My / 2 + 1 ? cuda::real_t(row) : (cuda::real_t(p.My) - cuda::real_t(row))) / p.length_y;
    cuda::real_t k2 = kx * kx + ky * ky;

    printf("----d_integrate_stiff_debug:\n");
    printf("p.Nx21 = %d, p.My = %d\n", p.Nx21, p.My);
    printf("p.level = %d, tlev = %d, kx = %5.3f, ky = %5.3f\n", p.level, tlev, kx, ky);
    printf("row = %d, col = %d\n", row, col);
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev + 1, idx, A[p.level - tlev + 1][idx].re(), A[p.level - tlev + 1][idx].im());
    printf("delta_t = %f, diff = %f hv = %f, kx = %f, ky = %f\n", p.delta_t, p.diff, p.hv, kx, ky);
    cuda::cmplx_t sum_alpha(0.0, 0.0);
    cuda::cmplx_t sum_beta(0.0, 0.0);
    cuda::real_t temp_div = 1. / (cuda::ss3_alpha_d[tlev - 2][0] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));
    cuda::cmplx_t result(0.0, 0.0);

    printf("sum_alpha = (%f, %f)\n", sum_alpha.re(), sum_alpha.im());
    printf("sum_beta = (%f, %f)\n", sum_beta.re(), sum_beta.im());
    for(uint k = 1; k < tlev; k++)
    {
        printf("\ttlev=%d, k = %d\t %f * A[%d] + dt * %f * A_R[%d]\n", tlev, k, cuda::ss3_alpha_d[tlev - 2][tlev - k], p.level - k, cuda::ss3_beta_d[tlev - 2][tlev - k - 1], p.level - 1 - k);
        printf("\ttlev=%d, k = %d\t sum_alpha = (%f, %f) + %f * (%f, %f)\n", tlev, k, sum_alpha.re(), sum_alpha.im(), cuda::ss3_alpha_d[tlev - 2][tlev - k], (A[p.level -k][idx]).re(), (A[p.level -k][idx]).im());
        printf("\ttlev=%d, k = %d\t sum_beta = (%f, %f) + %f * (%f, %f)\n", tlev, k, sum_beta.re(), sum_beta.im(), cuda::ss3_beta_d[tlev - 2][tlev - k - 1], (A_rhs[p.level - 1 - k][idx]).re(), (A_rhs[p.level - 1 - k][idx]).im());
        sum_alpha += A[p.level - k][idx] * cuda::ss3_alpha_d[tlev - 2][tlev - k];
        sum_beta += A_rhs[p.level - 1 - k][idx] * cuda::ss3_beta_d[tlev - 2][tlev - k - 1];
    }
    result = (sum_alpha + (sum_beta * p.delta_t)) * temp_div;
    printf("\ttlev=%d, computing A[%d], gamma_0 = %f\n", tlev, p.level - tlev, cuda::ss3_alpha_d[tlev - 2][0]);
    printf("sum1_alpha = (%f, %f)\tsum1_beta = (%f, %f)\t", sum_alpha.re(), sum_alpha.im(), sum_beta.re(),  sum_beta.im());
    printf("temp_div = %f\n", temp_div); 
    printf("A[%d][%d] = (%f, %f)\n", p.level - tlev, idx, result.re(), result.im());
    printf("\n");
}


/********************************************************************************
 *
 * Kernels to compute non-linear operators
 *
 ********************************************************************************/


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

    if((row < My) && (col < Nx))
    	out[idx] = f_x[idx] * g_y[idx] - f_y[idx] * g_x[idx];
    return;
}


// RHS for logarithmic density field:
// theta_x * strmf_y - theta_y * strmf_x + diff * (theta_x^2 + theta_y^2)
__global__
void d_theta_rhs_log(cuda::real_t* theta_x, cuda::real_t* theta_y, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t diff, cuda::real_t* res, const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    if((row < My) && (col < Nx))
    	res[idx] = theta_x[idx] * strmf_y[idx] - theta_y[idx] * strmf_x[idx] + diff * (theta_x[idx] * theta_x[idx] + theta_y[idx] * theta_y[idx]);
    return;
}


__global__
void d_theta_rhs_hw(cuda::cmplx_t* theta_rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, cuda::cmplx_t* strmf_y_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    if((row < My) && (col < Nx21))
    	theta_rhs_hat[idx] += ((strmf_hat[idx] - theta_hat[idx]) * C) - strmf_y_hat[idx];
    return;
}


__global__
void d_theta_sheath_nlin(cuda::real_t* res, cuda::real_t* theta_x, cuda::real_t* theta_y,
                         cuda::real_t* strmf, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t* tau,
                         const cuda::real_t alpha, const cuda::real_t delta, const cuda::real_t diff, 
                         const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    const double expT = exp(tau[idx]);
    if ((row < My) && (col < Nx))
        res[idx] = theta_x[idx] * strmf_y[idx] - theta_y[idx] * strmf_x[idx]
                   - alpha * sqrt(expT) * exp(twodads::Sigma - delta * strmf[idx] / expT)
                   + diff * (theta_x[idx] * theta_x[idx] + theta_y[idx] * theta_y[idx]);
    return;
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
void d_omega_ic(cuda::real_t* res, cuda::real_t* omega, cuda::real_t* omega_x, cuda::real_t* omega_y,
                cuda::real_t* strmf, 
                cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t* theta_y,
                const cuda::real_t ic, const cuda::real_t sdiss, const cuda::real_t cfric, 
                const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    if((row < My) && (col < Nx))
        res[idx] = omega_x[idx] * strmf_y[idx] - omega_y[idx] * strmf_x[idx] - theta_y[idx] * ic + strmf[idx] * sdiss + omega[idx] * cfric;
    return;
}


__global__
void d_omega_sheath_nlin(cuda::real_t* result, cuda::real_t* strmf, cuda::real_t* strmf_x, cuda::real_t* strmf_y, 
                         cuda::real_t* omega_x, cuda::real_t* omega_y, cuda::real_t* tau, cuda::real_t* tau_y,
                         cuda::real_t* theta_y, 
                         const cuda::real_t beta, const cuda::real_t delta, const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    const double expT = exp(tau[idx]);
    if((row < My) && (col < Nx))
    	result[idx] = omega_x[idx] * strmf_y[idx] - omega_y[idx] * strmf_x[idx] 
                      + beta * (1.0 - exp(twodads::Sigma - delta * strmf[idx] / expT))
                      - expT * (theta_y[idx] + tau_y[idx]);
    return;
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

    if((row < My) && (col < Nx21))
    	omega_rhs_hat[idx] += (strmf_hat[idx] - theta_hat[idx]) * C;
}


__global__
void d_tau_sheath_nlin(cuda::real_t* rhs, cuda::real_t* tau_x, cuda::real_t* tau_y,
                       cuda::real_t* strmf, cuda::real_t* strmf_x, cuda::real_t* strmf_y, cuda::real_t* tau,
                       const cuda::real_t alpha, const cuda::real_t delta, const cuda::real_t diff, 
                       const uint My, const uint Nx)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx + col;

    const double expT = exp(tau[idx]);
    if ((row < My) && (col < Nx))
        rhs[idx] = tau_x[idx] * strmf_y[idx] - tau_y[idx] * strmf_x[idx]
                   - 5.5 * alpha * sqrt(expT) * exp(twodads::Sigma - delta * strmf[idx] / expT)
                   + diff * (tau_x[idx] * tau_x[idx] + tau_y[idx] * tau_y[idx]);
    return;
}
 

__global__
void d_coupling_hwmod(cuda::cmplx_t* rhs_hat, cuda::cmplx_t* strmf_hat, cuda::cmplx_t* theta_hat, const cuda::real_t C, const uint My, const uint Nx21)
{
    // Start row with offset 1, this skips all ky=0 modes
    const uint row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;

    if((row < My) && (col < Nx21))
    	rhs_hat[idx] += (strmf_hat[idx] - theta_hat[idx]) * C;
    return;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

map<twodads::rhs_t, slab_cuda::rhs_fun_ptr> slab_cuda :: rhs_func_map = slab_cuda::create_rhs_func_map();

/// Initializes all real and fourier fields
/// Sets pointer to RHS functions for theta and omega
slab_cuda :: slab_cuda(const slab_config& my_config) :
    config(my_config),
    My(my_config.get_my()),
    Nx(my_config.get_nx()),
    tlevs(my_config.get_tlevs()),
    theta(1, My, Nx), theta_x(1, My, Nx), theta_y(1, My, Nx),
    tau  (1, My, Nx), tau_x  (1, My, Nx), tau_y  (1, My, Nx),
    omega(1, My, Nx), omega_x(1, My, Nx), omega_y(1, My, Nx),
    strmf(1, My, Nx), strmf_x(1, My, Nx), strmf_y(1, My, Nx),
    tmp_array(1, My, Nx), tmp_x_array(1, My, Nx), tmp_y_array(1, My, Nx), 
    theta_rhs(1, My, Nx), tau_rhs(1, My, Nx), omega_rhs(1, My, Nx),
    theta_hat(tlevs, My, Nx / 2 + 1), theta_x_hat(1, My, Nx / 2 + 1), theta_y_hat(1, My, Nx / 2 + 1),
    tau_hat  (tlevs, My, Nx / 2 + 1), tau_x_hat  (1, My, Nx / 2 + 1), tau_y_hat  (1, My, Nx / 2 + 1),
    omega_hat(tlevs, My, Nx / 2 + 1), omega_x_hat(1, My, Nx / 2 + 1), omega_y_hat(1, My, Nx / 2 + 1),
    strmf_hat(1,     My, Nx / 2 + 1), strmf_x_hat(1, My, Nx / 2 + 1), strmf_y_hat(1, My, Nx / 2 + 1),
    tmp_array_hat(1, My, Nx / 2 + 1), 
    theta_rhs_hat(tlevs - 1, My, Nx / 2 + 1),
    tau_rhs_hat  (tlevs - 1, My, Nx / 2 + 1),
    omega_rhs_hat(tlevs - 1, My, Nx / 2 + 1),
    dft_is_initialized(init_dft()),
    stiff_params(config.get_deltat(), config.get_lengthx(), config.get_lengthy(), config.get_model_params(0),
            config.get_model_params(1), My, Nx / 2 + 1, tlevs),
    slab_layout(config.get_xleft(), config.get_deltax(), config.get_ylow(), config.get_deltay(), config.get_deltat(), My, Nx),
    der(slab_layout),
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
    grid_dy_single(dim3(((Nx / 2) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, 1)),
    get_field_by_name{ {twodads::field_t::f_theta,     &theta},
	                   {twodads::field_t::f_theta_x,   &theta_x},
	                   {twodads::field_t::f_theta_y,   &theta_y},
	                   {twodads::field_t::f_tau,       &tau},
	                   {twodads::field_t::f_tau_x,     &tau_x},
	                   {twodads::field_t::f_tau_y,     &tau_y},
	                   {twodads::field_t::f_omega,     &omega},
	                   {twodads::field_t::f_omega_x,   &omega_x},
	                   {twodads::field_t::f_omega_y,   &omega_y},
	                   {twodads::field_t::f_strmf,     &strmf},
	                   {twodads::field_t::f_strmf_x,   &strmf_x},
	                   {twodads::field_t::f_strmf_y,   &strmf_y},
	                   {twodads::field_t::f_tmp,       &tmp_array},
	                   {twodads::field_t::f_theta_rhs, &theta_rhs},
	                   {twodads::field_t::f_tau_rhs,   &tau_rhs},
	                   {twodads::field_t::f_omega_rhs, &omega_rhs}},
	get_field_k_by_name{ {twodads::field_k_t::f_theta_hat,      &theta_hat},
		                 {twodads::field_k_t::f_theta_x_hat,    &theta_x_hat},
		                 {twodads::field_k_t::f_theta_y_hat,    &theta_y_hat},
		                 {twodads::field_k_t::f_tau_hat,        &tau_hat},
		                 {twodads::field_k_t::f_tau_x_hat,      &tau_x_hat},
		                 {twodads::field_k_t::f_tau_y_hat,      &tau_y_hat},
		                 {twodads::field_k_t::f_omega_hat,      &omega_hat},
		                 {twodads::field_k_t::f_omega_x_hat,    &omega_x_hat},
		                 {twodads::field_k_t::f_omega_y_hat,    &omega_y_hat},
		                 {twodads::field_k_t::f_strmf_hat,      &strmf_hat},
		                 {twodads::field_k_t::f_strmf_x_hat,    &strmf_x_hat},
		                 {twodads::field_k_t::f_strmf_y_hat,    &strmf_y_hat},
		                 {twodads::field_k_t::f_theta_rhs_hat,  &theta_rhs_hat},
		                 {twodads::field_k_t::f_tau_rhs_hat,    &tau_rhs_hat},
		                 {twodads::field_k_t::f_omega_rhs_hat,  &omega_rhs_hat},
		                 {twodads::field_k_t::f_tmp_hat,        &tmp_array_hat}},
    get_output_by_name{ {twodads::output_t::o_theta,     &theta},
		                {twodads::output_t::o_theta_x,   &theta_x},
		                {twodads::output_t::o_theta_y,   &theta_y},
		                {twodads::output_t::o_tau,       &tau},
		                {twodads::output_t::o_tau_x,     &tau_x},
		                {twodads::output_t::o_tau_y,     &tau_y},
		                {twodads::output_t::o_omega,     &omega},
		                {twodads::output_t::o_omega_x,   &omega_x},
		                {twodads::output_t::o_omega_y,   &omega_y},
		                {twodads::output_t::o_strmf,     &strmf},
		                {twodads::output_t::o_strmf_x,   &strmf_x},
		                {twodads::output_t::o_strmf_y,   &strmf_y},
		                {twodads::output_t::o_theta_rhs, &theta_rhs},
		                {twodads::output_t::o_tau_rhs, &tau_rhs},
		                {twodads::output_t::o_omega_rhs, &omega_rhs}},
    rhs_array_map{ {twodads::field_k_t::f_theta_hat, &theta_rhs_hat},
                   {twodads::field_k_t::f_tau_hat, &tau_rhs_hat},
		           {twodads::field_k_t::f_omega_hat, &omega_rhs_hat}}
{
    theta_rhs_func = rhs_func_map[config.get_theta_rhs_type()];
    tau_rhs_func = rhs_func_map[config.get_tau_rhs_type()];
    omega_rhs_func = rhs_func_map[config.get_omega_rhs_type()];

#ifdef DEBUG
    print_address();
#endif // DEBUG
    gpuStatus();
}


slab_cuda :: ~slab_cuda()
{
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

void slab_cuda :: initialize()
{
    // In this switch block we compute all initial fields (theta, omega and/or strmf)
    // After this block the slab is in the following state:
    // * theta, tau, omega and/or strmf are in the desired initial state
    // * theta_hat, tau_hat, omega_hat are initialized for tlev - 1
    // * strmf_hat is initialized at tlev=0 (since it is not a dynamic field)
    // * spatial derivatives have not been computed 
    // * RHS has not been evaluated
    switch(config.get_init_function_theta())
    {
        case twodads::init_fun_t::init_gaussian:
            init_gaussian(&theta, config.get_initc_theta(), slab_layout, config.get_log_theta());
            dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, config.get_tlevs() - 1);
            break;
    
        case twodads::init_fun_t::init_constant:
            if(config.get_log_theta())
                theta = log(config.get_initc_theta(0));
            else
                theta = config.get_initc_theta(0);

            dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_sine:
            init_simple_sine(&theta, config.get_initc_theta(), slab_layout);
            dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_mode:
            init_mode(&theta_hat, config.get_initc_theta(), slab_layout, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_turbulent_bath:
        	init_turbulent_bath(&theta_hat, slab_layout, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_lamb_dipole:
            theta_hat = cuda::cmplx_t(0.0, 0.0);
            break;
        default:
        	throw config_error(string("init function: ") + config.get_init_function_theta_str() + string(" is mapped to init_NA\n"));
    }

    switch(config.get_init_function_omega())
    {
        case twodads::init_fun_t::init_gaussian:
            init_gaussian(&omega, config.get_initc_omega(), slab_layout, config.get_log_omega());
            dft_r2c(twodads::field_t::f_omega, twodads::field_k_t::f_omega_hat, config.get_tlevs() - 1);
            break;
    
        case twodads::init_fun_t::init_constant:
            omega = config.get_initc_omega(0);
            dft_r2c(twodads::field_t::f_omega, twodads::field_k_t::f_omega_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_sine:
            init_simple_sine(&omega, config.get_initc_omega(), slab_layout);
            dft_r2c(twodads::field_t::f_omega, twodads::field_k_t::f_omega_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_mode:
            init_mode(&omega_hat, config.get_initc_omega(), slab_layout, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_turbulent_bath:
        	init_turbulent_bath(&omega_hat, slab_layout, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_lamb_dipole:
            init_lamb(&omega, config.get_initc_omega(), slab_layout);
            dft_r2c(twodads::field_t::f_omega, twodads::field_k_t::f_omega_hat, config.get_tlevs() - 1);
            inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, config.get_tlevs() - 1);
            break;
        default:
        	throw config_error(string("init function: ") + config.get_init_function_omega_str() + string(" is mapped to init_NA\n"));

    }

    switch(config.get_init_function_tau())
    {
        case twodads::init_fun_t::init_gaussian:
            init_gaussian(&tau, config.get_initc_tau(), slab_layout, config.get_log_tau());
            dft_r2c(twodads::field_t::f_tau, twodads::field_k_t::f_tau_hat, config.get_tlevs() - 1);
            break;
    
        case twodads::init_fun_t::init_constant:
            if(config.get_log_tau())
                tau = log(config.get_initc_tau(0));
            else
                tau = config.get_initc_tau(0);
            dft_r2c(twodads::field_t::f_tau, twodads::field_k_t::f_tau_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_sine:
            init_simple_sine(&tau, config.get_initc_tau(), slab_layout);
            dft_r2c(twodads::field_t::f_tau, twodads::field_k_t::f_tau_hat, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_mode:
            init_mode(&tau_hat, config.get_initc_tau(), slab_layout, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_turbulent_bath:
        	init_turbulent_bath(&tau_hat, slab_layout, config.get_tlevs() - 1);
            break;

        case twodads::init_fun_t::init_lamb_dipole:
            tau_hat = cuda::cmplx_t(0.0, 0.0);
            break;

        default:
        	throw config_error(string("init function: ") + config.get_init_function_tau_str() + string(" is mapped to init_NA\n"));
    }

    // Compute spatial derivatives and RHS
    d_dx_dy(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_x_hat, twodads::field_k_t::f_theta_y_hat, config.get_tlevs() - 1);
    d_dx_dy(twodads::field_k_t::f_tau_hat, twodads::field_k_t::f_tau_x_hat, twodads::field_k_t::f_tau_y_hat, config.get_tlevs() - 1);
    d_dx_dy(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_omega_x_hat, twodads::field_k_t::f_omega_y_hat, config.get_tlevs() - 1);
    d_dx_dy(twodads::field_k_t::f_strmf_hat, twodads::field_k_t::f_strmf_x_hat, twodads::field_k_t::f_strmf_y_hat, 0);

    dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, config.get_tlevs() - 1);
    dft_c2r(twodads::field_k_t::f_theta_x_hat, twodads::field_t::f_theta_x, 0);
    dft_c2r(twodads::field_k_t::f_theta_y_hat, twodads::field_t::f_theta_y, 0);

    dft_c2r(twodads::field_k_t::f_tau_hat, twodads::field_t::f_tau, config.get_tlevs() - 1);
    dft_c2r(twodads::field_k_t::f_tau_x_hat, twodads::field_t::f_tau_x, 0);
    dft_c2r(twodads::field_k_t::f_tau_y_hat, twodads::field_t::f_tau_y, 0);

    dft_c2r(twodads::field_k_t::f_omega_hat, twodads::field_t::f_omega, config.get_tlevs() - 1);
    dft_c2r(twodads::field_k_t::f_omega_x_hat, twodads::field_t::f_omega_x, 0);
    dft_c2r(twodads::field_k_t::f_omega_y_hat, twodads::field_t::f_omega_y, 0);

    dft_c2r(twodads::field_k_t::f_strmf_hat, twodads::field_t::f_strmf, 0);
    dft_c2r(twodads::field_k_t::f_strmf_x_hat, twodads::field_t::f_strmf_x, 0);
    dft_c2r(twodads::field_k_t::f_strmf_y_hat, twodads::field_t::f_strmf_y, 0);

    // Compute RHS from last time levels
    rhs_fun(config.get_tlevs() - 1);
    move_t(twodads::field_k_t::f_theta_rhs_hat, config.get_tlevs() - 2, 0);
    move_t(twodads::field_k_t::f_tau_rhs_hat, config.get_tlevs() - 2, 0);
    move_t(twodads::field_k_t::f_omega_rhs_hat, config.get_tlevs() - 2, 0);
}


// Move data from time level t_src to t_dst
void slab_cuda :: move_t(const twodads::field_k_t fname, const uint t_dst, const uint t_src)
{
    cuda_arr_cmplx* arr = get_field_k_by_name.at(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> move(t_dst, t_src);
}


// Copy data from time level t_src to t_dst
void slab_cuda :: copy_t(const twodads::field_k_t fname, const uint t_dst, const uint t_src)
{
    cuda_arr_cmplx* arr = get_field_k_by_name.at(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> copy(t_dst, t_src);
}


// Set fname to a constant value at time index tlev
void slab_cuda::set_t(const twodads::field_k_t fname, const cuda::cmplx_t val, const uint t_src)
{
    cuda_arr_cmplx* arr = get_field_k_by_name.at(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    //arr -> set_t(val, t_src);
    arr -> op_scalar_t<d_op1_assign<cuda::cmplx_t> >(val, t_src);
}


/// Set fname to constant value at time index tlev=0
void slab_cuda::set_t(const twodads::field_t fname, const cuda::real_t val, const uint t_src)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
#ifdef DEBUG
    gpuStatus();
#endif
    arr -> op_scalar_t<d_op1_assign<cuda::real_t> >(val, t_src);
}

/// advance all fields with multiple time levels
void slab_cuda :: advance()
{
    theta_hat.advance();
    theta_rhs_hat.advance();
    tau_hat.advance();
    tau_rhs_hat.advance();
    omega_hat.advance();
    omega_rhs_hat.advance();
}


/// Compute RHS from using time index t_src for dynamical fields omega_hat and theta_hat.
void slab_cuda :: rhs_fun(const uint t_src)
{
    (this ->* theta_rhs_func)(t_src);
    (this ->* tau_rhs_func)(t_src);
    (this ->* omega_rhs_func)(t_src);
}


// Update real fields theta, theta_x, theta_y, etc.
void slab_cuda::update_real_fields(const uint tlev, const uint t)
{
    //Compute theta, omega, strmf and respective spatial derivatives

    d_dx_dy(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_x_hat, twodads::field_k_t::f_theta_y_hat, tlev);
    dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, tlev);
    dft_c2r(twodads::field_k_t::f_theta_x_hat, twodads::field_t::f_theta_x, 0);
    dft_c2r(twodads::field_k_t::f_theta_y_hat, twodads::field_t::f_theta_y, 0);
    
    d_dx_dy(twodads::field_k_t::f_tau_hat, twodads::field_k_t::f_tau_x_hat, twodads::field_k_t::f_tau_y_hat, tlev);
    dft_c2r(twodads::field_k_t::f_tau_hat, twodads::field_t::f_tau, tlev);
    dft_c2r(twodads::field_k_t::f_tau_x_hat, twodads::field_t::f_tau_x, 0);
    dft_c2r(twodads::field_k_t::f_tau_y_hat, twodads::field_t::f_tau_y, 0);
    
    d_dx_dy(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_omega_x_hat, twodads::field_k_t::f_omega_y_hat, tlev);
    dft_c2r(twodads::field_k_t::f_omega_hat, twodads::field_t::f_omega, tlev);
    dft_c2r(twodads::field_k_t::f_omega_x_hat, twodads::field_t::f_omega_x, 0);
    dft_c2r(twodads::field_k_t::f_omega_y_hat, twodads::field_t::f_omega_y, 0);
    
    //inv_laplace(omega_hat, strmf_hat, tlev);
    //inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, tlev);
    d_dx_dy(twodads::field_k_t::f_strmf_hat, twodads::field_k_t::f_strmf_x_hat, twodads::field_k_t::f_strmf_y_hat, 0);
    dft_c2r(twodads::field_k_t::f_strmf_hat, twodads::field_t::f_strmf, 0);
    dft_c2r(twodads::field_k_t::f_strmf_x_hat, twodads::field_t::f_strmf_x, 0);
    dft_c2r(twodads::field_k_t::f_strmf_y_hat, twodads::field_t::f_strmf_y, 0);

    dft_c2r(twodads::field_k_t::f_theta_rhs_hat, twodads::field_t::f_theta_rhs, 1);
    dft_c2r(twodads::field_k_t::f_tau_rhs_hat, twodads::field_t::f_tau_rhs, 1);
    dft_c2r(twodads::field_k_t::f_omega_rhs_hat, twodads::field_t::f_omega_rhs, 1);
}


void slab_cuda::copy_device_to_host(const twodads::output_t f_name)
{
    cuda_arr_real* arr_r = get_output_by_name.at(f_name);
    arr_r -> copy_device_to_host();
}

// execute r2c DFT
void slab_cuda :: dft_r2c(const twodads::field_t fname_r, const twodads::field_k_t fname_c, const uint t_dst)
{
    cuda_arr_real* arr_r = get_field_by_name.at(fname_r);
    cuda_arr_cmplx* arr_c = get_field_k_by_name.at(fname_c);
#ifdef DEBUG
    bool terminate_after_check = false;
    cerr << "slab_cuda :: dft_r2c" << endl;
    cerr << "\rreal data at " << arr_r -> get_array_d() << "\tcomplex data at " << arr_c -> get_array_d(t_dst) << "\tt_dst = " << t_dst << endl;

    try
    {
#endif

    der.dft_r2c(arr_r -> get_array_d(), arr_c -> get_array_d(t_dst));

#ifdef DEBUG
    }
    catch(assert_error err)
    {
        // Parseval's theorem was not fulfilled in derivatives. Check it again here ->
        // Write out fields and exit
        terminate_after_check = true;
    }
    cerr << "slab_cuda :: dft_r2c" << endl;
    arr_r -> copy_device_to_host();
    arr_c -> copy_device_to_host();

    cuda::real_t sum_r2 = 0.0;
    cuda::real_t sum_c2 = 0.0;

    ofstream of_r;
    ofstream of_c;
    if(terminate_after_check)
    {   
        of_r.open("slabcuda_arr_r.dat", ios::trunc);
        of_c.open("slabcuda_arr_c.dat", ios::trunc);
    }

    for(uint m = 0; m < My; m++)
    {
        for(uint n = 0; n < Nx; n++)
        {
            sum_r2 += (*arr_r)(0, m, n) * (*arr_r)(0, m, n);
            // Take hermitian symmetry into consideration
            if(n < Nx / 2 + 1)
                sum_c2 += (*arr_c)(t_dst, m, n).abs() * (*arr_c)(t_dst, m, n).abs();
            else
                sum_c2 += (*arr_c)(t_dst, (My - m) % My, Nx - n).abs() * (*arr_c)(t_dst, (My - m) % My, Nx - n).abs();


            if(terminate_after_check)
            {
                of_r << (*arr_r)(0, m, n) << "\t";
                if(n < Nx / 2 + 1)
                    of_c << (*arr_c)(t_dst, m, n) << "\t";
            }
        }
        if(terminate_after_check)
        {
            of_r << "\n";
            of_c << "\n";
        }
    }
    sum_c2 = sum_c2 / cuda::real_t(My * Nx);

    if(terminate_after_check)
    {
        of_r.close();
        of_c.close();
    }

    if(abs(sum_r2) > 1e-10 && abs(sum_c2) > 1e-10)
    {
        cuda::real_t rel_err = (abs(sum_r2 - sum_c2) / abs(sum_r2));
        cerr << "   slab_cuda :: dft_r2c: sum_r2 = " << sum_r2 << "\tsum_c2 / (N*M)= " << sum_c2 << "\tRel. err: " << rel_err << endl;
        if(rel_err > 1e-8)
        {
            ofstream of;
            of.open("arr_r.dat", ios::trunc);
            of << (*arr_r);
            of.close();
            of.open("arr_c.dat", ios::trunc);
            of << (*arr_c);
            of.close();
        }
        assert(rel_err < 1e-8);
    }
    if(terminate_after_check)
        throw assert_error("Parseval;s theorem differis in dft_r2c in derivs and slab_cuda!");
    cerr << "============================================================================================================\n\n\n\n";
#endif

}


// execute iDFT and normalize the resulting real field
void slab_cuda :: dft_c2r(const twodads::field_k_t fname_c, const twodads::field_t fname_r, const uint t_src)
{
    cuda_arr_cmplx* arr_c = get_field_k_by_name.at(fname_c);
    cuda_arr_real* arr_r = get_field_by_name.at(fname_r);
#ifdef DEBUG
    bool terminate_after_check = false;
    cerr << "slab_cuda :: dft_c2r" << endl;
    cerr << "\tcomplex data at " << arr_c -> get_array_d(t_src) << "\treal data at " << arr_r -> get_array_d() << "\tt_src = " << t_src << endl;

    try
    {
#endif // DEBUG
        der.dft_c2r(arr_c -> get_array_d(t_src), arr_r -> get_array_d());
        arr_r -> normalize();
#ifdef DEBUG
    }
    catch(assert_error err)
    {
        // If Parseval's theorem was not fulfilled in derivatives, check it again here ->
        // Write out fields and exit
        terminate_after_check = true;   
    }

    cerr << "slab_cuda :: dft_c2r" << endl;
    arr_r -> copy_device_to_host();
    arr_c -> copy_device_to_host();
    cuda::real_t sum_r2 = 0.0;
    cuda::real_t sum_c2 = 0.0;

    ofstream of_r;
    ofstream of_c;
    if (terminate_after_check)
    {
        of_r.open("slabcuda_arr_r.dat", ios::trunc);
        of_c.open("slabcuda_arr_c.dat", ios::trunc);
    }

    for(uint m = 0; m < My; m++)
    {
        for(uint n = 0; n < Nx; n++)
        {
            sum_r2 += (*arr_r)(0, m, n) * (*arr_r)(0, m, n);

            if(n < Nx / 2 + 1)
                sum_c2 += (*arr_c)(t_src, m, n).abs() * (*arr_c)(t_src, m, n).abs();
            else
                sum_c2 += (*arr_c)(t_src, (My - m) % My, Nx - n).abs() * (*arr_c)(t_src, (My - m) % My, Nx - n).abs();

            if(terminate_after_check)
            {
                of_r << (*arr_r)(0, m, n) << "\t";
                if( n < Nx / 2 + 1)
                    of_c << (*arr_c)(t_src, m, n) << "\t";
            }
        }
        if (terminate_after_check)
        {
            of_r << "\n";
            of_c << "\n";
        }
    }
    sum_c2 = sum_c2 / cuda::real_t(My * Nx);
   
    if(terminate_after_check)
    {
        of_r.close();
        of_c.close();
    }

    if(abs(sum_r2) > 1e-10 && abs(sum_c2) > 1e-10)
    {
        cuda::real_t rel_err = (abs(sum_r2 - sum_c2) / abs(sum_r2));

        cerr << "   slab_cuda :: dft_c2r: sum_r2 = " << sum_r2 << "\tsum_c2 / (N*M)= " << sum_c2 << "\tRel. err: " << rel_err << endl;
        if(rel_err > 1e-8)
        {
            ofstream of;
            of.open("arr_r.dat", ios::trunc);
            of << (*arr_r);
            of.close();
            of.open("arr_c.dat", ios::trunc);
            of << (*arr_c);
            of.close();
        }
        assert(rel_err < 1e-8);
    }


    if(terminate_after_check)
        throw assert_error("Parseval's theorem differs in dft_c2r in slab_cuda and derivs!");
    cerr << "============================================================================================================\n\n\n\n";
#endif
}


// print real field on terminal
void slab_cuda :: print_field(const twodads::field_t field_name) const
{
    cout << *get_field_by_name.at(field_name) << endl;
}

// print real field to ascii file
void slab_cuda :: print_field(const twodads::field_t field_name, const string file_name) const
{
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_by_name.at(field_name) << "\n";
    output_file.close();
}


// print complex field on terminal, all time levels
void slab_cuda :: print_field(const twodads::field_k_t field_name) const
{
    cout << *get_field_k_by_name.at(field_name) << endl;
}


// print complex field to ascii file
void slab_cuda :: print_field(const twodads::field_k_t field_name, const string file_name) const
{
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_k_by_name.at(field_name) << "\n";
    output_file.close();
}


/// Copy data from a cuda_array<cuda::real_t> to a cuda::real_t* buffer, tlev=0
void slab_cuda :: get_data_host(const twodads::field_t fname, cuda_arr_real& buffer) const
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    arr -> check_bounds(0, buffer.get_my(), buffer.get_nx());
    arr -> copy_device_to_host(buffer.get_array_h(0));
}

/// Copy data from a field_t into an external buffer
void slab_cuda :: get_data_host(const twodads::field_t fname, cuda::real_t* buffer, const uint My, const uint Nx) const
{
	cuda_arr_real* arr = get_field_by_name.at(fname);
	arr -> check_bounds(0, My, Nx);
	arr -> copy_device_to_host(buffer);
}


/// Copy device data from a field_t into an external buffer
void slab_cuda :: get_data_device(const twodads::field_t fname, cuda::real_t* buffer, const uint My, const uint Nx) const
{
	cuda_arr_real* arr = get_field_by_name.at(fname);
	arr -> check_bounds(0, My, Nx);
	arr -> copy_device_to_device(0, buffer);
}

/// Update device data and return a pointer to requested array
cuda_arr_real* slab_cuda :: get_array_ptr(const twodads::output_t fname)
{
    cuda_arr_real* arr = get_output_by_name.at(fname);
    arr -> copy_device_to_host();
    return arr;
}


/// Update device data and return a pointer to requested array
cuda_arr_real* slab_cuda :: get_array_ptr(const twodads::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    arr -> copy_device_to_host();
    return arr;
}


void slab_cuda :: print_address() const
{
    // Use this to test of memory is aligned between g++ and NVCC
    cout << "slab_cuda::print_address()\n";

    cout << "theta_hat at " << (void*) &theta_hat;
    for(uint t = 0; t < 4; t++)
        cout << "\tt=" << t << " at " << theta_hat.get_array_d(t) << "\t";
    cout << endl;
    cout << "theta_x_hat at " << (void*) &theta_x_hat << "\t, data at " << theta_x_hat.get_array_d() << endl;
    cout << "theta_y_hat at " << (void*) &theta_y_hat << "\t, data at " << theta_y_hat.get_array_d() << endl;

    cout << "tau_hat at " << (void*) &tau_hat;
    for(uint t = 0; t < 4; t++)
        cout << "\tt=" << t << " at " << tau_hat.get_array_d(t) << "\t";
    cout << endl;
    cout << "tau_x_hat at " << (void*) &tau_x_hat << "\t, data at " << tau_x_hat.get_array_d() << endl;
    cout << "tau_y_hat at " << (void*) &tau_y_hat << "\t, data at " << tau_y_hat.get_array_d() << endl;

    cout << "omega_hat at " << (void*) &omega_hat;
    for(uint t = 0; t < 4; t++)
        cout << "\tt=" << t << " at " << omega_hat.get_array_d(t) << "\t";
    cout << endl;
    cout << "omega_x_hat at " << (void*) &omega_x_hat << "\t, data at " << omega_x_hat.get_array_d() << endl;
    cout << "omega_y_hat at " << (void*) &omega_y_hat << "\t, data at " << omega_y_hat.get_array_d() << endl;

    cout << "theta at " << (void*) &theta << "\t, data at " << theta.get_array_d() << endl;
    cout << "theta_x at " << (void*) &theta_x << "\t, data at " << theta_x.get_array_d() << endl;
    cout << "theta_y at " << (void*) &theta_y << "\t, data at " << theta_y.get_array_d() << endl;

    cout << "tau at " << (void*) &tau << "\t, data at " << tau.get_array_d() << endl;
    cout << "tau_x at " << (void*) &tau_x << "\t, data at " << tau_x.get_array_d() << endl;
    cout << "tau_y at " << (void*) &tau_y << "\t, data at " << tau_y.get_array_d() << endl;

    cout << "omega at " << (void*) &omega << "\t, data at " << omega.get_array_d() << endl;
    cout << "omega_x at " << (void*) &omega_x << "\t, data at " << omega_x.get_array_d() << endl;
    cout << "omega_y at " << (void*) &omega_y << "\t, data at " << omega_y.get_array_d() << endl;

    cout << "strmf at " << (void*) &strmf << "\t, data at " << strmf.get_array_d() << endl;
    cout << "strmf_x at " << (void*) &strmf_x << "\t, data at " << strmf_x.get_array_d() << endl;
    cout << "strmf_y at " << (void*) &strmf_y << "\t, data at " << strmf_y.get_array_d() << endl;
}


void slab_cuda :: print_grids() const
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


/*****************************************************************************
 *
 * Function implementation
 *
 ****************************************************************************/


void slab_cuda :: enumerate(const twodads::field_k_t f_name)
{
    get_field_k_by_name.at(f_name) -> enumerate_array(0);
}


void slab_cuda :: enumerate(const twodads::field_t f_name)
{
    get_field_by_name.at(f_name) -> enumerate_array(0);
}


void slab_cuda :: d_dx_dy(const twodads::field_k_t src_name, const twodads::field_k_t dst_x_name,
		const twodads::field_k_t dst_y_name, const uint t_src)
{
	cuda_arr_cmplx* arr_in = get_field_k_by_name.at(src_name);
    cuda_arr_cmplx* arr_x = get_field_k_by_name.at(dst_x_name);
	cuda_arr_cmplx* arr_y = get_field_k_by_name.at(dst_y_name);

    der.d_dx1_dy1(arr_in, arr_x, arr_y, t_src);
}


// Invert laplace operator in fourier space, using src field at time index t_src, store result in dst_name, time index 0
void slab_cuda :: inv_laplace(const twodads::field_k_t src_name, const twodads::field_k_t dst_name, const uint t_src)
{
    cuda_arr_cmplx* arr_in = get_field_k_by_name.at(src_name);
    cuda_arr_cmplx* arr_out = get_field_k_by_name.at(dst_name);

    der.inv_laplace(arr_in, arr_out, t_src);
}


void slab_cuda :: integrate_stiff(const twodads::field_k_t fname, const uint tlev)
{
    cuda_arr_cmplx* A = get_field_k_by_name.at(fname);
    cuda_arr_cmplx* A_rhs = rhs_array_map.at(fname);

    //d_integrate_stiff_debug<<<1, 1>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), d_ss3_alpha, d_ss3_beta, stiff_params, tlev, 4, 4);
    switch(tlev)
    {
     case 2:
    	d_integrate_stiff_map_2<<<grid_my_nx21, block_my_nx21>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), stiff_params);
        break;
    case 3:
    	d_integrate_stiff_map_3<<<grid_my_nx21, block_my_nx21>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), stiff_params);
    	break;
    case 4:
    	d_integrate_stiff_map_4<<<grid_my_nx21, block_my_nx21>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), stiff_params);
    	break;
    }
    gpuStatus();
}

/// @brief Integrate only modes with ky=0
void slab_cuda :: integrate_stiff_ky0(const twodads::field_k_t fname, const uint tlev)
{
    cuda_arr_cmplx* A = get_field_k_by_name.at(fname);
    cuda_arr_cmplx* A_rhs = rhs_array_map.at(fname);

    d_integrate_stiff_ky0<<<grid_dy_single, block_nx21>>>(A->get_array_d_t(), A_rhs->get_array_d_t(), stiff_params, tlev);
    gpuStatus();
}


void slab_cuda :: integrate_stiff_debug(const twodads::field_k_t fname, const uint tlev, const uint row, const uint col)
{
    cuda_arr_cmplx* A = get_field_k_by_name.at(fname);
    cuda_arr_cmplx* A_rhs = rhs_array_map.at(fname);

    d_integrate_stiff_debug<<<1, 1>>>(A -> get_array_d_t(), A_rhs -> get_array_d_t(), stiff_params, tlev, row, col);
    gpuStatus();
}


/// Set explicit part for theta equation to zero
void slab_cuda :: theta_rhs_null(const uint t)
{
    theta_rhs_hat.op_scalar_t<d_op1_assign<cuda::cmplx_t> >(cuda::cmplx_t(0.0, 0.0), 0);
}


// Compute RHS for Navier stokes model, store result in time index 0 of theta_rhs_hat
void slab_cuda :: theta_rhs_ns(const uint t_src)
{
    //cout << "theta_rhs_ns\n";
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_theta_rhs_hat, 0);
}


void slab_cuda :: theta_rhs_lin(const uint t_src)
{
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();

    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_theta_rhs_hat, 0);
}


//Compute explicit part for Hasegawa-Wakatani model, store result in time index 0 of theta_rhs_hat
void slab_cuda :: theta_rhs_hw(const uint t_src)
{
    static const cuda::real_t C = config.get_model_params(2);
    // Poisson bracket is on the RHS: dn/dt = {n, phi} + ...
    // Zero out RHS
    tmp_array.op_scalar_t<d_op1_assign<cuda::real_t> >(0.0, 0);

    // Comupute poisson bracket
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_tmp_hat, 0);
    //Copy poisson bracket to new RHS
    d_theta_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(tmp_array_hat.get_array_d(), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), strmf_y_hat.get_array_d(), C, My, Nx / 2 + 1);
    gpuStatus();

    theta_rhs_hat.op_array_t<d_op1_assign<cuda::cmplx_t> >(tmp_array_hat, 0);
}


//Explicit part for the MHW model
void slab_cuda :: theta_rhs_hwmod(const uint t_src)
{
    static const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_theta_rhs_hat, 0);
    // Neglect ky=0 modes for in coupling term
    d_coupling_hwmod<<<grid_my_nx21, block_my_nx21>>>(theta_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
    gpuStatus();
    theta_rhs_hat -= strmf_y_hat; 

    gpuStatus();
}


void slab_cuda :: theta_rhs_log(const uint t_src)
{
    d_theta_rhs_log<<<grid_my_nx, block_my_nx>>>(theta_x.get_array_d(), theta_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), stiff_params.diff, tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_theta_rhs_hat, 0);
}


void slab_cuda :: theta_rhs_sheath_nlin(const uint t_src)
{
    static const cuda::real_t alpha = config.get_model_params(2);
    static const cuda::real_t delta = config.get_model_params(4);
    static const cuda::real_t diff = config.get_model_params(0);

    d_theta_sheath_nlin<<<grid_my_nx, block_my_nx>>>(tmp_array.get_array_d(0), theta_x.get_array_d(0), theta_y.get_array_d(0),
                                                     strmf.get_array_d(0), strmf_x.get_array_d(0), strmf_y.get_array_d(0), tau.get_array_d(0),
                                                     alpha, delta, diff, My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_theta_rhs_hat, 0);
}


//Set explicit part for omega equation to zero
void slab_cuda :: omega_rhs_null(const uint t)
{
    omega_rhs_hat.op_scalar_t<d_op1_assign<cuda::cmplx_t> >(cuda::cmplx_t(0.0, 0.0), 0);
}


/// Explicit part for Navier-Stokes model
void slab_cuda :: omega_rhs_ns(const uint t_src)
{
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    //gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_omega_rhs_hat, 0);

    //static cuda_array<cuda::real_t> pb1(1, My, Nx);
    //static cuda_array<cuda::real_t> pb2(1, My, Nx);
    //static cuda_array<cuda::real_t> pb3(1, My, Nx);

    /*
     * Method 1:
     * {f, g} = f_x g_y - f_y g_x
     */
    // pb1 = omega_x * strmf_y - omega_y * strmf_x;

    /*
     * Method 2:
     * {f, g} = (f g_y)_x - (f g_x)_y
     */ 
    //tmp_array = omega * strmf_y;

    //der.d_dx1_dy1(tmp_array, pb2, tmp_y_array);
   
    //tmp_array = omega * strmf_x;
    //der.d_dx1_dy1(tmp_array, tmp_x_array, tmp_y_array);
    //pb2 -= tmp_y_array;
    //der.dft_r2c(pb2.get_array_d(), omega_rhs_hat.get_array_d(0));

    /*
     * Method 3:
     * {f, g} = (f_x g)_y - (f_y g)_x
     */

    //tmp_array = omega_x * strmf;
    //der.d_dx1_dy1(tmp_array, tmp_x_array, pb3);

    //tmp_array = omega_y * strmf;
    //der.d_dx1_dy1(tmp_array, tmp_x_array, tmp_y_array);
    //pb3 -= tmp_x_array;
    //der.dft_r2c(pb2.get_array_d(), omega_rhs_hat.get_array_d(0));


    //pb1 += pb2;
    //pb1 += pb3;
    //pb1 /= 0.3;

    //der.dft_r2c(pb3.get_array_d(), omega_rhs_hat.get_array_d(0));
}

/// RHS for the Hasegawa-Wakatani model
void slab_cuda :: omega_rhs_hw(const uint t_src)
{
    static const cuda::real_t C = config.get_model_params(2);
    tmp_array.op_scalar_t<d_op1_assign<cuda::real_t> >(0.0, 0);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_tmp_hat, 0);

    d_omega_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(tmp_array_hat.get_array_d(), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
    gpuStatus();

    omega_rhs_hat.op_array_t<d_op1_assign<cuda::cmplx_t> >(tmp_array_hat, 0);
}

/// RHS for modified Hasegawa-Wakatani model
void slab_cuda :: omega_rhs_hwmod(const uint t_src)
{
    static const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_omega_rhs_hat, 0);
    gpuStatus();
    d_coupling_hwmod<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
}


//RHS for modified Hasegawa-Wakatani model, remove zonal flows
void slab_cuda :: omega_rhs_hwzf(const uint t_src)
{
    static const cuda::real_t C = config.get_model_params(2);
    d_pbracket<<<grid_my_nx, block_my_nx>>>(omega_x.get_array_d(), omega_y.get_array_d(), strmf_x.get_array_d(), strmf_y.get_array_d(), tmp_array.get_array_d(), My, Nx);
    gpuStatus();
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_omega_rhs_hat, 0);
    d_kill_ky0<<<grid_ky0, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), My, Nx / 2 + 1);
    gpuStatus();
    d_omega_rhs_hw<<<grid_my_nx21, block_my_nx21>>>(omega_rhs_hat.get_array_d(0), strmf_hat.get_array_d(), theta_hat.get_array_d(t_src), C, My, Nx / 2 + 1);
    gpuStatus();
}

void slab_cuda::omega_rhs_ic(const uint t_src)
{
    // Convert model parameters to complex numbers
    static const cuda::real_t ic = config.get_model_params(2); 
    static const cuda::real_t sdiss = config.get_model_params(3);
    static const cuda::real_t cfric = config.get_model_params(4);

#ifdef DEBUG
    cout << "omega_rhs_ic" << endl;
    cout << "ic = " << ic << ", sdiss = " << sdiss << ", cfric = " << cfric << endl;
    cout << "grid = (" << omega_hat.get_grid().x << ", " << omega_hat.get_grid().y << "), block = (" << omega_hat.get_block().x << ", " << omega_hat.get_block().y << ")" << endl;
#endif //DEBUG
    d_omega_ic<<<grid_my_nx, block_my_nx>>>(tmp_array.get_array_d(), omega.get_array_d(), omega_x.get_array_d(), 
                                            omega_y.get_array_d(), strmf.get_array_d(), strmf_x.get_array_d(), 
                                            strmf_y.get_array_d(), theta_y.get_array_d(),
                                            ic, sdiss, cfric, My, Nx);
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_omega_rhs_hat, 0);
    gpuStatus();
}


void slab_cuda :: omega_rhs_sheath_nlin(const uint t_src)
{
    static const cuda::real_t beta = config.get_model_params(3);
    static const cuda::real_t delta = config.get_model_params(4);

#ifdef DEBUG
    cout << "omega_rhs_sheath_nlin " << endl;
    cout << "beta = " << beta << ", delta = " << delta << endl;
#endif //DEBUG
    d_omega_sheath_nlin<<<grid_my_nx, block_my_nx>>>(tmp_array.get_array_d(0), strmf.get_array_d(0), strmf_x.get_array_d(0), strmf_y.get_array_d(0),
                                                     omega_x.get_array_d(0), omega_y.get_array_d(0), tau.get_array_d(0), 
                                                     tau_y.get_array_d(0), theta_y.get_array_d(0),
                                                     beta, delta, My, Nx);
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_omega_rhs_hat, 0);
}


void slab_cuda :: tau_rhs_sheath_nlin(const uint t_src)
{
    static const cuda::real_t alpha = config.get_model_params(2);
    static const cuda::real_t delta = config.get_model_params(4);
    static const cuda::real_t diff = config.get_model_params(0);

#ifdef DEBUG
    cout << "tau_rhs_sheath_nlin " << endl;
    cout << "alpha = " << alpha << ", delta = " << delta << ", diff = " << diff << endl;
#endif //DEBUG
    d_tau_sheath_nlin<<<grid_my_nx, block_my_nx>>>(tmp_array.get_array_d(0), tau_x.get_array_d(0), tau_y.get_array_d(0),
                                                     strmf.get_array_d(0), strmf_x.get_array_d(0), strmf_y.get_array_d(0), tau.get_array_d(0), 
                                                     alpha, delta, diff, My, Nx);
    dft_r2c(twodads::field_t::f_tmp, twodads::field_k_t::f_tau_rhs_hat, 0);
}


void slab_cuda :: tau_rhs_null(const uint t_src)
{
    tau_rhs_hat.op_scalar_t<d_op1_assign<cuda::cmplx_t> >(cuda::cmplx_t(0.0, 0.0), 0);
}

// End of file slab_cuda.cu
