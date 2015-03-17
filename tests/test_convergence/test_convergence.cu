/*
 * Test convergence of poisson brackets
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cuda_array4.h>
#include <cuda_darray.h>

#ifdef __CUDA_ARCH__
#define CUDAMEMBER __device__ 
#endif

#ifndef __CUDA_ARCH__
#define CUDAMEMBER
#endif

using namespace std;


template <int MY, int NX21>
__global__
void gen_k_map_dx1_dy1(cuda::cmplx_t* kmap, const cuda::real_t two_pi_Lx, const cuda::real_t two_pi_Ly)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * NX21 + col;

    cuda::cmplx_t tmp(0.0, 0.0);

    if(row < MY / 2)
        tmp.set_im(two_pi_Ly * cuda::real_t(row));
    else if (row == MY / 2)
        tmp.set_im(0.0);
    else
        tmp.set_im(two_pi_Ly * (cuda::real_t(row) - cuda::real_t(MY)));

    if(col < NX21 - 1)
        tmp.set_re(two_pi_Lx * cuda::real_t(col));
    else
        tmp.set_re(0.0);

    if((col < NX21) && (row < MY))
        kmap[index] = tmp;
}

template <int MY, int NX21, int T>
__global__
void d_dx_dy_map(cuda::cmplx_t*  in, cuda::cmplx_t*  out_x, cuda::cmplx_t*  out_y, cuda::cmplx_t*  kmap)
{
	// offset to index for current row.
	const int row_offset = (blockIdx.y * blockDim.y + threadIdx.y) * NX21;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row_offset + col;
	int t;

	for(t = 0; t < T; t++)
	{
		if(col < NX21)
		{
			index = row_offset + col;
			out_x[index] = in[index] * cuda::cmplx_t(0.0, kmap[index].re());
			out_y[index] = in[index] * cuda::cmplx_t(0.0, kmap[index].im());
		}
		else
			break;
		col++;
		index++;
	}
}


#define XNULL 0.1
#define YNULL 0.1

template <typename T>
class fun1 {
    // f(x, y) = exp(-(x - x_0)^2 - (y - y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};

template <typename T>
class fun1_x {
    // f_x(x, y) = -2 (x - x_0) exp(-(x - x_0)^2 - (y - y_0)^2)
    public: 
        CUDAMEMBER T operator() (T x, T y) const { return( -2.0 * (x - XNULL) * exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};

template <typename T>
class fun1_y {
    // f_y(x, y) = -2 (y - y_0) exp(-(x - x_0)^2 - (y - y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return( -2.0 * (y - YNULL) * exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};

template <typename T>
class fun2 {
    // g(x, y) = exp(-(x + x_0)^2 - (y + y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( exp(-(x + XNULL) * (x + XNULL) - (y + YNULL) * (y + YNULL)) );};
};

template <typename T>
class fun2_x {
    // g_x(x, y) = -2 (x + x_0) exp(-(x + x_0)^2 - (y + y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( -2.0 * (x + XNULL) * exp(-(x + XNULL) * (x + XNULL) - (y + YNULL) * (y + YNULL)) );};
};

template <typename T>
class fun2_y {
    // g_y(x, y) = -2 (y + y_0) exp(-(x + x_0)^2 - (y + y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( -2.0 * (y + YNULL) * exp(-(x + XNULL) * (x + XNULL) - (y + YNULL) * (y + YNULL)) );};
};


template <typename T>
class poisson {
    public:
        CUDAMEMBER T operator() (T x, T y) const { return( exp(-2.0 * (x * x + XNULL * XNULL + y * y + YNULL * YNULL)) *(-8. * XNULL * y + 8. * YNULL * x));};
};

int main(void)
{
    cufftHandle plan_r2c;
    cufftHandle plan_c2r;

    //int Nx, My;
    //cout << "Enter Nx: " << endl;
    //cin >> Nx;
    //cout << "Enter My: " << endl;
    //cin >> My;

    constexpr int Nx{512};
    constexpr int My{512};
    constexpr int Nx21{Nx / 2 + 1};
    constexpr cuda::real_t Lx{10.0};
    constexpr cuda::real_t Ly{10.0};
    constexpr cuda::real_t dx{Lx / Nx};
    constexpr cuda::real_t dy{Lx / My};

    constexpr cuda::real_t two_pi_Lx{cuda::TWOPI / Lx};
    constexpr cuda::real_t two_pi_Ly{cuda::TWOPI / Ly};

    constexpr int elem_per_thread{1};

    constexpr int blocksize_nx{64};
    constexpr int blocksize_my{4};

	constexpr int num_block_x{(Nx21 + (elem_per_thread * blocksize_nx - 1)) / (elem_per_thread * blocksize_nx)};
	constexpr int num_block_y{(My + blocksize_my - 1) / blocksize_my};
	dim3 blocksize(blocksize_nx, blocksize_my);
	dim3 gridsize(num_block_x, num_block_y);

    cout << "Lx = " << Lx << ", Nx = " << Nx << ", dx = " << dx << ", dy = " << dy << endl;
    cout << "blocksize = (" << blocksize.x << ", " << blocksize.y << ")" << endl;
    cout << "gridsize = (" << gridsize.x << ", " << gridsize.y << ")" << endl;

    const cuda::slab_layout_t sl(-0.5 * Lx, Lx / (cuda::real_t) Nx, -0.5 * Ly, Ly / (cuda::real_t) My, My, Nx);

    ofstream of;

    // Plan DFTs
    cufftResult err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z);
    if (err != CUFFT_SUCCESS)
        throw;
    err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D);
    if (err != CUFFT_SUCCESS)
        throw;

    /* Define f, f_x, f_y, ... in real and fourier space*/
    cuda_array<cuda::real_t> f_r(1, My, Nx);
    cuda_array<cuda::real_t> fx_r(1, My, Nx);
    cuda_array<cuda::real_t> fy_r(1, My, Nx);

    cuda_array<cuda::real_t> fx_analytic(1, My, Nx);
    cuda_array<cuda::real_t> fy_analytic(1, My, Nx);

    cuda_array<cuda::cmplx_t> fhat(1, My, Nx21);
    cuda_array<cuda::cmplx_t> fx_hat(1, My, Nx21);
    cuda_array<cuda::cmplx_t> fy_hat(1, My, Nx21);

    cuda_array<cuda::real_t> g_r(1, My, Nx);
    cuda_array<cuda::real_t> gx_r(1, My, Nx);
    cuda_array<cuda::real_t> gy_r(1, My, Nx);

    cuda_array<cuda::real_t> gx_analytic(1, My, Nx);
    cuda_array<cuda::real_t> gy_analytic(1, My, Nx);

    cuda_array<cuda::cmplx_t> ghat(1, My, Nx21);
    cuda_array<cuda::cmplx_t> gx_hat(1, My, Nx21);
    cuda_array<cuda::cmplx_t> gy_hat(1, My, Nx21);

    /* Numerical computed poisson bracket */
    cuda_array<cuda::real_t> sol_num(1, My, Nx);
    /* Analytic solution of the poisson bracket */
    cuda_array<cuda::real_t> sol_an(1, My, Nx);

    cuda_array<cuda::cmplx_t> k_map(1, My, Nx21);

    gen_k_map_dx1_dy1<My, Nx21> <<<dim3(1, My), dim3(Nx21, 1)>>>(k_map.get_array_d(), two_pi_Lx, two_pi_Ly);
	gpuErrchk(cudaPeekAtLastError());
    gpuStatus();

    //cout << "kmap = " << k_map;

    /* Evaluate f and g */
    f_r.op_scalar_fun<fun1<cuda::real_t> >(sl, 0);
    of.open("f.dat");
    of << f_r;
    of.close();

    g_r.op_scalar_fun<fun2<cuda::real_t> >(sl, 0);
    of.open("g.dat");
    of << g_r;
    of.close();

    sol_an.op_scalar_fun<poisson<cuda::real_t> >(sl, 0);
    of.open("sol_an.dat");
    of << sol_an;
    of.close();

    /* Transform to fourier space */
    err = cufftExecD2Z(plan_r2c, f_r.get_array_d(), (cufftDoubleComplex*) fhat.get_array_d(0));
    if (err != CUFFT_SUCCESS)
        throw;

    err = cufftExecD2Z(plan_r2c, g_r.get_array_d(), (cufftDoubleComplex*) ghat.get_array_d(0));
    if (err != CUFFT_SUCCESS)
        throw;

    /* compute derivatives */
    d_dx_dy_map<My, Nx21, elem_per_thread><<<blocksize, gridsize>>>(fhat.get_array_d(0), fx_hat.get_array_d(), fy_hat.get_array_d(), k_map.get_array_d());
	gpuErrchk(cudaPeekAtLastError());
    gpuStatus();

    d_dx_dy_map<My, Nx21, elem_per_thread><<<blocksize, gridsize>>>(ghat.get_array_d(0), gx_hat.get_array_d(), gy_hat.get_array_d(), k_map.get_array_d());
	gpuErrchk(cudaPeekAtLastError());
    gpuStatus();

    ///* Transform to real space */
    cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) fx_hat.get_array_d(0), fx_r.get_array_d(0));
    if (err != CUFFT_SUCCESS)
        throw;
    fx_r.normalize();

    err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) fy_hat.get_array_d(0), fy_r.get_array_d(0));
    if (err != CUFFT_SUCCESS)
        throw;
    fy_r.normalize();

    err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) gx_hat.get_array_d(0), gx_r.get_array_d(0));
    if (err != CUFFT_SUCCESS)
        throw;
    gx_r.normalize();
    err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) gy_hat.get_array_d(0), gy_r.get_array_d(0));
    if (err != CUFFT_SUCCESS)
        throw;
    gy_r.normalize();

    // Compare numerical derivatives to analytic expressions
    fx_analytic.op_scalar_fun<fun1_x<cuda::real_t> >(sl, 0);
    fy_analytic.op_scalar_fun<fun1_y<cuda::real_t> >(sl, 0);

    gx_analytic.op_scalar_fun<fun2_x<cuda::real_t> >(sl, 0);
    gy_analytic.op_scalar_fun<fun2_y<cuda::real_t> >(sl, 0);


    of.open("f_x.dat");
    of << fx_r;
    of.close();

    of.open("f_y.dat");
    of << fy_r;
    of.close();

    of.open("g_x.dat");
    of << gx_r;
    of.close();

    of.open("g_y.dat");
    of << gy_r;
    of.close();


    // compute L2 error of derivatives
    //cout << "fx_r:  blocksize = (" << fx_r.get_block().x << ", " << fx_r.get_block().y << ")" << endl;
    //cout << "        gridsize = (" << fx_r.get_grid().x  << ", " << fx_r.get_grid().y  << ")" << endl;
    cuda_darray<cuda::real_t> diff_deriv(My, Nx);
    diff_deriv = (fx_r - fx_analytic) * (fx_r - fx_analytic);
    //diff_deriv = 0.0;
    //cout << "diff_deriv:  blocksize = (" << static_cast<cuda_array<cuda::real_t>*>(&diff_deriv) -> get_block().x << ", " << static_cast<cuda_array<cuda::real_t>* >(&diff_deriv) ->  get_block().y << ")" << endl;
    //cout << "              gridsize = (" << static_cast<cuda_array<cuda::real_t>*>(&diff_deriv) -> get_grid().x  << ", " << static_cast<cuda_array<cuda::real_t>*>(&diff_deriv) -> get_grid().y  << ")" << endl;
    cout << "L2 norm of error in f_x is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;
    diff_deriv = (fy_r - fy_analytic) * (fy_r - fy_analytic);
    cout << "L2 norm of error in f_y is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl; 
    diff_deriv = (gx_r - gx_analytic) * (gx_r - gx_analytic);
    cout << "L2 norm of error in g_x is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;
    diff_deriv = (gy_r - gy_analytic) * (gy_r - gy_analytic);
    cout << "L2 norm of error in g_y is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;

    ///* Compute poisson bracket */
    sol_num = fx_r * gy_r - fy_r * gx_r;
    diff_deriv = (sol_num - sol_an) * (sol_num - sol_an);
    cout << "L2 norm of error in poisson bracket is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;

    of.open("sol_num.dat");   
    of << sol_num;
    of.close();

    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
    return(0);
}
