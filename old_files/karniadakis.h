/*
 * Implements stiffly-stable karniadakis scheme
 *
 */


#include "2dads_typs.h"
#include "solvers.h";
#include "slab_config.h"


#ifdef __CUDACC__
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
		temp_div = 1.0 / (cuda::ss3_alpha_d[0][0] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));


		dummy = A[3][idx];
		A[2][idx] = (A[3][idx] + (A_rhs[2][idx] * cuda::ss3_beta_d[0][0] * p.delta_t)) * temp_div;
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
		temp_div = 1. / (cuda::ss3_alpha_d[1][0] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

		sum_alpha = (A[3][idx] * cuda::ss3_alpha_d[1][2]) + (A[2][idx] * cuda::ss3_alpha_d[1][1]);
		sum_beta = (A_rhs[2][idx] * cuda::ss3_beta_d[1][1]) + (A_rhs[1][idx] * cuda::ss3_beta_d[1][0]);
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
		temp_div = 1. / (cuda::ss3_alpha_d[2][0] + p.delta_t * (p.diff * k2 + p.hv * k2 * k2 * k2));

		sum_alpha = A[3][idx] * cuda::ss3_alpha_d[2][3] + A[2][idx] * cuda::ss3_alpha_d[2][2] + A[1][idx] * cuda::ss3_alpha_d[2][1];
		sum_beta = A_rhs[2][idx] * cuda::ss3_beta_d[2][2] + A_rhs[1][idx] * cuda::ss3_beta_d[2][1] + A_rhs[0][idx] * cuda::ss3_beta_d[2][0];
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

	sum_alpha = A[3][idx] * cuda::ss3_alpha_d[2][3] + A[2][idx] * cuda::ss3_alpha_d[2][2] + A[1][idx] * cuda::ss3_alpha_d[2][1];
	sum_beta = A_rhs[2][idx] * cuda::ss3_beta_d[2][2] + A_rhs[1][idx] * cuda::ss3_beta_d[2][1] + A_rhs[0][idx] * cuda::ss3_beta_d[2][0];

	printf("\td_integrate_stiff_map_4_debug\n");
	printf("\tsum_alpha = (%f, %f) * %f\n", A[3][idx].re(), A[3][idx].im(), cuda::ss3_alpha_d[2][3]);
	printf("\t          + (%f, %f) * %f\n", A[2][idx].re(), A[2][idx].im(), cuda::ss3_alpha_d[2][2]);
	printf("\t          + (%f, %f) * %f\n", A[1][idx].re(), A[1][idx].im(), cuda::ss3_alpha_d[2][1]);
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




#endif // __CUDACC__


template <typename T>
class karniadakis
{
    public:
        karniadakis(const slab_config);
        ~karniadakis();


        void stiff_2(CuCmplx<T>**, CuCmplx<T>**);

        void stiff_3(CuCmplx<T>**, CuCmplx<T>**);

        void stiff_4(CuCmplx<T>**, CuCmplx<T>**);


    private:
        slab_layout_t sl;
        stiff_params_t p;

};


template <typename T>
karniadakis<T> :: karniadakis(const slab_config cfg)
{
    cout << "karniadakis :: karniadakis" << endl;
    cout << p << endl;

}



