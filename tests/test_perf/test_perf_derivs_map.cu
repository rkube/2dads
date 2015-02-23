/*
 * test_perf_derivs.cu
 *
 *  Created on: Feb 12, 2015
 *      Author: rku000
 */


/*
 * Test different ways of computing derivation of a field
 *
 *
 * ---------> x direction (Nx/2+1 modes)
 * |
 * |
 * |
 * V
 * y-direction My modes (My / 2 positive, My / 2 - 1 negative
 *
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include "cucmplx.h"
#include "cuda_array4.h"
#include "cufft.h"

typedef CuCmplx<double> cmplx_t;

#define ELEM_PER_THREAD_T 1


using namespace std;

/*
 * Method 1:
 *
 *Compute x, and y derivative in one pass. Use a highly divergent kernel
 *
 */

__global__
void d_dx_dy_div(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y,
                double two_pi_Lx, double two_pi_Ly, const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    cuda::cmplx_t ikx(0.0, 0.0);
    cuda::cmplx_t iky(0.0, 0.0);

    // Set to unity to enumerate modes
    two_pi_Lx = 1.0;
    two_pi_Ly = 1.0;
    // block d_dy_lo
    if(row < My / 2)
    {
        iky.set_im(two_pi_Ly * double(row));
        if(col < Nx21 - 1)
            ikx.set_im(two_pi_Lx * double(col));
        // skip col == Nx21 block, as ikx is initialized with 0.0
    }
    else if(row == My / 2)
    {
        // iky is 0.0 here
        if(col < Nx21 - 1)
            ikx.set_im(two_pi_Lx * double(col));
    }
    else if (row > My / 2)
    {
        iky.set_im(two_pi_Ly * (double(My) - double(col)));
    if(col < Nx21 - 1)
        ikx.set_im(two_pi_Lx * double(col));
    }

    if((row < My) && (col < Nx21))
    {
        out_x[index] = ikx;
        out_y[index] = iky;
    }
}



/*
 * Method 2
 *
 * Compute array with ikx, iky
 *
 * Store them interleaved, i.e.
 *
 * CuCmplx<double> k
 * k.re() = kx
 * k.im() = ky
 *
 *
 */


template <int MY, int NX21>
__global__
void gen_k_map(cmplx_t* kmap, const double two_pi_Lx, const double two_pi_Ly)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * NX21 + col;

    cmplx_t tmp(0.0, 0.0);

    if(row < MY / 2)
        tmp.set_im(two_pi_Ly * double(row));
    else if (row == MY / 2)
        tmp.set_im(0.0);
    else
        tmp.set_im(two_pi_Ly * ((double(row - MY))));

    if(col < NX21 - 1)
        tmp.set_re(two_pi_Lx * double(col));
    else
        tmp.set_re(0.0);

    if((col < NX21) && (row < MY))
        kmap[index] = tmp;
}



template <int MY, int NX21>
__global__
void gen_k_map_enum(cmplx_t* kmap)
{

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * NX21 + col;

    cmplx_t tmp(0.0, 0.0);

    if(row < MY / 2)
        tmp.set_im(1.0);
    else if (row == MY / 2)
        tmp.set_im(2.0);
    else
        tmp.set_im(3.0);

    if(col < NX21 - 1)
        tmp.set_re(1.0);
    else
        tmp.set_re(2.0);

    if((col < NX21) && (row < MY))
        kmap[index] = tmp;
}




template <int MY, int NX21, int T>
__global__
void d_dx_dy_map(cmplx_t*  in, cmplx_t*  out_x, cmplx_t*  out_y, cmplx_t*  kmap)
{
	// offset to index for current row.
	const int row_offset = (blockIdx.y * blockDim.y + threadIdx.y) * NX21;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row_offset + col;
	int t;

#pragma unroll 4
	for(t = 0; t < T; t++)
	{
		if(col < NX21)
		{
			index = row_offset + col;
			out_x[index] = in[index] * cmplx_t(0.0, kmap[index].re());
			out_y[index] = in[index] * cmplx_t(0.0, kmap[index].im());
		}
		else
			break;
		col++;
		index++;
	}
}


// Same as above, but with shared memory
template <int MY, int NX21, int T>
__global__
void d_dx_dy_map_sh(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, cmplx_t* kmap)
{
	extern __shared__ cmplx_t shmem[];
	const int row_offset = (blockIdx.y * blockDim.y + threadIdx.y) * NX21;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int offset_s = 2 * threadIdx.x;
	int t;

#pragma unroll 1
	for(t = 0; t < T; t++)
	{
		if(col < NX21)
		{
			shmem[offset_s + 2 * t    ] = in[row_offset + col];
			shmem[offset_s + 2 * t + 1] = kmap[row_offset + col];
		}
		col++;
	}
	col = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll 1
	for(t = 0; t < T; t++)
	{
		if(col < NX21)
		{
			out_x[row_offset + col] = shmem[offset_s + 2 * t] * cmplx_t(0.0, shmem[offset_s + 2 * t + 1].re());
			out_y[row_offset + col] = shmem[offset_s + 2 * t] * cmplx_t(0.0, shmem[offset_s + 2 * t + 1].im());
		}
		col++;
	}
}




int main(void)
{
	constexpr int Nx{512};
	constexpr int Nx21{Nx / 2 + 1};
	constexpr int My{512};
	constexpr double Lx{10.0};
	constexpr double Ly{10.0};
	constexpr double dx{Lx / Nx};
	constexpr double dy{Ly / My};
	constexpr double two_pi_Lx{2.0 * 3.1415926 / Lx};
	constexpr double two_pi_Ly{2.0 * 3.1415926 / Ly};


	constexpr int blocksize_nx{32};
	constexpr int blocksize_my{1};

	const int elem_per_thread{ELEM_PER_THREAD_T};

	// d_dy_dx_map kernel loads kmap and input array into shared memory
	const size_t shmem_size = 2 * elem_per_thread * blocksize_nx * sizeof(cmplx_t);

	constexpr int num_block_x{(Nx21 + (elem_per_thread * blocksize_nx - 1)) / (elem_per_thread * blocksize_nx)};

	constexpr int num_block_y{(My + blocksize_my - 1) / blocksize_my};
	dim3 blocksize(blocksize_nx, blocksize_my);
	dim3 gridsize(num_block_x, num_block_y);

	cout << "blocksize: (" << blocksize.x << ", " << blocksize.y << ")" << endl;
	cout << "gridsize : (" << gridsize.x  << ", " << gridsize.y  << ")" << endl;


	const cmplx_t foo(0.0, 0.0);

	cuda_array<double> r_arr(1, My, Nx);
	cuda_array<double> r_arr_x(1, My, Nx);
	cuda_array<double> r_arr_y(1, My, Nx);

	cuda_array<cmplx_t> c_arr(1, My, Nx / 2 + 1);
	cuda_array<cmplx_t> c_arr_x(1, My, Nx / 2 + 1);
	cuda_array<cmplx_t> c_arr_y(1, My, Nx / 2 + 1);

	cuda_array<cmplx_t> kmap(1, My, Nx / 2 + 1);

	double x{0.0};
	double y{0.0};
	for(int m = 0; m < My; m++)
	{
		y = - 0.5 * Ly + m * dy;
		for(int n = 0; n < Nx; n++)
		{
			x = -0.5 * Lx + n * dx;
			r_arr(0, m, n) = exp(-0.5 * x * x - 0.5 * y * y);
		}
	}
	r_arr.copy_host_to_device();

	// Initialize cufft
	cufftResult err;
    cufftHandle plan_r2c;
    cufftHandle plan_c2r;
    err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z);
    err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D);

	err = cufftExecD2Z(plan_r2c, r_arr.get_array_d(), (cufftDoubleComplex*) c_arr.get_array_d(0));
	if(err != CUFFT_SUCCESS)
		throw;

	// Method 1... generate k-map and run derivs in one kernel call
	gen_k_map<My, Nx21><<<dim3(1, My), dim3(Nx21, 1)>>>(kmap.get_array_d(), two_pi_Lx, two_pi_Ly);
	gpuErrchk(cudaPeekAtLastError());

//	void d_dx_dy_map(cmplx_t*  in, cmplx_t*  out_x, cmplx_t*  out_y, cmplx_t*  kmap)

	//d_dx_dy_map<My, Nx21, elem_per_thread><<<gridsize, blocksize>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), kmap.get_array_d());
	d_dx_dy_map_sh<My, Nx21, elem_per_thread><<<gridsize, blocksize, shmem_size>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), kmap.get_array_d());

	gpuErrchk(cudaPeekAtLastError());


	// Transform to real space
	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr.get_array_d(), r_arr.get_array_d());
	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr_x.get_array_d(), r_arr_x.get_array_d());
	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr_y.get_array_d(), r_arr_y.get_array_d());

	r_arr.normalize();
	r_arr_x.normalize();
	r_arr_y.normalize();


	// output
	ofstream of;
	of.open("r_arr.dat");
	of << r_arr;
	of.close();

	of.open("r_arr_x.dat");
	of << r_arr_x;
	of.close();

	of.open("r_arr_y.dat");
	of << r_arr_y;
	of.close();


	for(int t = 0; t < 1000; t++)
	{
	    d_dx_dy_map_sh<My, Nx21, elem_per_thread><<<gridsize, blocksize, shmem_size>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), kmap.get_array_d());
		if (t % 50 == 0)
			cout << t << endl;
	}
}



