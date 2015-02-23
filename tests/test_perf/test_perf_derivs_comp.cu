/*
 * test_derivs_perf_comp.cu
 *
 *  Created on: Feb 12, 2015
 *      Author: rku000
 *
 * Test derivation with explicit computation of the multiplicators
 *
 * The idea is, that these can happen, while the kernel waits to get the input
 * and output data from memory
 *
 * Use sector based kernels, as to eliminate branching in kernels
 *
 * Use streams, to launch derivative kernels simultaneously
 *
 * Array boundaries templated into kernels
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

using namespace std;

// Enum kernels to test block and grid sizes
template <int MY, int NX21>
__global__
void d_dy_dx_11_enum(cmplx_t* in)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * NX21 + col;
	if(col < NX21 - 1)
		in[index] = cmplx_t(1.0, 1.0);
}

// row < My / 2
// col < Nx / 2
template<int MY, int NX21>
__global__
void d_dy_dx_11(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, const double two_pi_Lx, const double two_pi_Ly)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * NX21 + col;

    // Return if we don't have an item to work on
    if(col < NX21 - 1)
    {
    	out_x[index] = in[index] * cmplx_t(0.0, two_pi_Lx * double(col));
    	out_y[index] = in[index] * cmplx_t(0.0, two_pi_Ly * double(row));
    }

    return;
}



template <int MY, int NX21>
__global__
void d_dy_dx_21_enum(cmplx_t* in)
{
	const int row = MY / 2;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * NX21 + col;
	if(col < NX21 - 1)
		in[index] = cmplx_t(2.0, 1.0);
}



// Frequencies: row = My / 2, stored in row My / 2
//              col < Nx / 2
template<int MY, int NX21>
__global__
void d_dy_dx_21(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, const double two_pi_Lx, const double two_pi_Ly)
{
    const int row = MY / 2;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * NX21 + col;

    if(col < NX21 - 1)
	{
		out_x[index] = in[index] * cmplx_t(0.0, two_pi_Lx * double(col));
		//out_y[index] = in[index];
        out_y[index].set_re(0.0);
        out_y[index].set_im(0.0);
	}
   	return;
}


template <int MY, int NX21>
__global__
void d_dy_dx_31_enum(cmplx_t* in)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y + MY / 2 + 1;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * NX21 + col;
	if(col < NX21 - 1)
		in[index] = cmplx_t(3.0, 1.0);
}



// Frequencies: row = My/2 + 1 ... My - 1
//              col < Nx / 2
// These are stored in the last My/2-1 rows
template<int MY, int NX21>
__global__
void d_dy_dx_31(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, const double two_pi_Lx, const double two_pi_Ly)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y + MY / 2 + 1;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * NX21 + col;
    if(col < NX21 - 1)
    {
    	out_x[index] = in[index] * cmplx_t(0.0, two_pi_Lx * double(col));
    	//out_y[index] = in[index];
        out_y[index] = in[index] * cmplx_t(0.0, two_pi_Ly * double(row - MY));
    }
    return;
}



// Frequencies row = 0 .. My / 2
//             col = Nx / 2
template <int MY, int NX21>
__global__
void d_dy_dx_12_enum(cmplx_t* in)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = NX21 - 1;
	const int index = row * NX21 + col;
	if(row < MY / 2)
		in[index] = cmplx_t(1.0, 2.0);
}


template<int MY, int NX21>
__global__
void d_dy_dx_12(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, const double two_pi_Lx, const double two_pi_Ly)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = NX21 - 1;
    const uint index = row * NX21 + col;

    if(row < MY / 2)
    {
        out_x[index].set_re(0.0);
        out_x[index].set_im(0.0);
        //out_y[index] = in[index];
        out_y[index] = in[index] * cmplx_t(0.0, two_pi_Ly * double(row));
    }
    return;
}


template <int MY, int NX21>
__global__
void d_dy_dx_22_enum(cmplx_t* in)
{
	const int row = MY / 2;
	const int col = NX21 - 1;
	const int index = row * NX21 + col;
	in[index] = cmplx_t(2.0, 2.0);
}



// Frequencies row = My / 2
//             col = Nx / 2
template<int MY, int NX21>
__global__
void d_dy_dx_22(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, const double two_pi_Lx, const double two_pi_Ly)
{
    const uint row = MY / 2;
    const uint col = NX21 - 1;
    const uint index = row * NX21 + col;

    out_x[index].set_re(0.0);
    out_x[index].set_im(0.0);
    //out_y[index] = in[index];
    out_y[index].set_re(0.0);
    out_y[index].set_im(0.0);
}

template <int MY, int NX21>
__global__
void d_dy_dx_32_enum(cmplx_t* in)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y + MY / 2 + 1;
	const int col = NX21 - 1;
	const int index = row * NX21 + col;
	if(row < MY)
		in[index] = cmplx_t(3.0, 2.0);

}

// Frequencies row = My / 2 + 1 ... My-1
//             col = Nx / 2
template<int MY, int NX21>
__global__
void d_dy_dx_32(cmplx_t* in, cmplx_t* out_x, cmplx_t* out_y, const double two_pi_Lx, const double two_pi_Ly)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y + MY / 2 + 1;
	const int col = NX21 - 1;
	const int index = row * NX21 + col;

	if(row < MY)
	{
		out_x[index].set_re(0.0);
		out_x[index].set_im(0.0);
		out_y[index] = in[index] * cmplx_t(0.0, two_pi_Ly * double(row - MY));
	}
}






int main(void)
{
	constexpr int Nx{256};
	constexpr int Nx21{Nx / 2 + 1};
	constexpr int My{256};
	constexpr double Lx{10.0};
	constexpr double Ly{10.0};
	constexpr double dx{Lx / Nx};
	constexpr double dy{Ly / My};
	constexpr double two_pi_Lx{2.0 * 3.1415926 / Lx};
	constexpr double two_pi_Ly{2.0 * 3.1415926 / Ly};


	constexpr int blocksize_nx{32};
	constexpr int num_blocks_x{ (Nx21 + blocksize_nx - 1) / blocksize_nx};
	constexpr int num_blocks_y{((My / 2 + 1) + blocksize_nx - 1) / blocksize_nx};

	constexpr int num_streams{6};
	cudaStream_t streams[num_streams];

	for(int s = 0; s < num_streams; s++)
	{
		cudaStreamCreate(&streams[s]);
	}

	dim3 bs_d_11(blocksize_nx, 1);
	dim3 gs_d_11(num_blocks_x, My / 2);
	cout << "bs_d_11 = (" << bs_d_11.x << ", " << bs_d_11.y << ")\t";
	cout << "gs_d_11 = (" << gs_d_11.x << ", " << gs_d_11.y << ")\n";

	dim3 bs_d_21(blocksize_nx, 1);
	dim3 gs_d_21(num_blocks_x, 1);
	cout << "bs_d_21 = (" << bs_d_21.x << ", " << bs_d_21.y << ")\t";
	cout << "gs_d_21 = (" << gs_d_21.x << ", " << gs_d_21.y << ")\n";

	dim3 bs_d_31(blocksize_nx, 1);
	dim3 gs_d_31(num_blocks_x, My / 2 - 1);
	cout << "bs_d_31 = (" << bs_d_31.x << ", " << bs_d_31.y << ")\t";
	cout << "gs_d_31 = (" << gs_d_31.x << ", " << gs_d_31.y << ")\n";

	dim3 bs_d_12(1, blocksize_nx);
	dim3 gs_d_12(1, num_blocks_y);
	cout << "bs_d_12 = (" << bs_d_12.x << ", " << bs_d_12.y << ")\t";
	cout << "gs_d_12 = (" << gs_d_12.x << ", " << gs_d_12.y << ")\n";

	dim3 bs_d_22(1, 1);
	dim3 gs_d_22(1, 1);
	cout << "bs_d_22 = (" << bs_d_22.x << ", " << bs_d_22.y << ")\t";
	cout << "gs_d_22 = (" << gs_d_22.x << ", " << gs_d_22.y << ")\n";

	dim3 bs_d_32(1, blocksize_nx);
	dim3 gs_d_32(1, num_blocks_y);
	cout << "bs_d_32 = (" << bs_d_32.x << ", " << bs_d_32.y << ")\t";
	cout << "gs_d_32 = (" << gs_d_32.x << ", " << gs_d_32.y << ")\n";



	cuda_array<double> r_arr(1, My, Nx);
	cuda_array<double> r_arr_x(1, My, Nx);
	cuda_array<double> r_arr_y(1, My, Nx);

	cuda_array<cmplx_t> c_arr(1, My, Nx / 2 + 1);
	cuda_array<cmplx_t> c_arr_x(1, My, Nx / 2 + 1);
	cuda_array<cmplx_t> c_arr_y(1, My, Nx / 2 + 1);

//	d_dy_dx_11_enum<My, Nx21><<<bs_d_11, gs_d_11>>>(c_arr.get_array_d());
//	gpuErrchk(cudaPeekAtLastError());
//	d_dy_dx_21_enum<My, Nx21><<<bs_d_21, gs_d_21>>>(c_arr.get_array_d());
//	d_dy_dx_31_enum<My, Nx21><<<bs_d_31, gs_d_31>>>(c_arr.get_array_d());
//	d_dy_dx_12_enum<My, Nx21><<<bs_d_12, gs_d_12>>>(c_arr.get_array_d());
//	d_dy_dx_22_enum<My, Nx21><<<bs_d_22, gs_d_22>>>(c_arr.get_array_d());
//	d_dy_dx_32_enum<My, Nx21><<<bs_d_32, gs_d_32>>>(c_arr.get_array_d());
//	ofstream of;
//	of.open("enum_out.dat");
//	of << c_arr;
//	of.close();




	double x{0.0};
	double y{0.0};
	for(int m = 0; m < My; m++)
	{
		y = - Ly/2 + m * dy;
		for(int n = 0; n < Nx; n++)
		{
			x = -Lx/2 + n * dx;
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

	d_dy_dx_11<My, Nx21><<<bs_d_11, gs_d_11>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
	gpuErrchk(cudaPeekAtLastError());
	d_dy_dx_21<My, Nx21><<<bs_d_21, gs_d_21>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
	d_dy_dx_31<My, Nx21><<<bs_d_31, gs_d_31>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
	d_dy_dx_12<My, Nx21><<<bs_d_12, gs_d_12>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
	d_dy_dx_22<My, Nx21><<<bs_d_22, gs_d_22>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
	d_dy_dx_32<My, Nx21><<<bs_d_32, gs_d_32>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);


	// Transform to real space
	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr.get_array_d(), r_arr.get_array_d());
	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr_x.get_array_d(), r_arr_x.get_array_d());
	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr_y.get_array_d(), r_arr_y.get_array_d());

	r_arr.normalize();
	r_arr_x.normalize();
	r_arr_y.normalize();

	r_arr_x.copy_device_to_host();
	r_arr_y.copy_device_to_host();

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
		d_dy_dx_11<My, Nx21><<<bs_d_11, gs_d_11, 0, streams[0]>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
		d_dy_dx_21<My, Nx21><<<bs_d_21, gs_d_21, 0, streams[0]>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
		d_dy_dx_31<My, Nx21><<<bs_d_31, gs_d_31, 0, streams[1]>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
		d_dy_dx_12<My, Nx21><<<bs_d_12, gs_d_12, 0, streams[1]>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
		d_dy_dx_22<My, Nx21><<<bs_d_22, gs_d_22, 0, streams[0]>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
		d_dy_dx_32<My, Nx21><<<bs_d_32, gs_d_32, 0, streams[1]>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), two_pi_Lx, two_pi_Ly);
		if(t % 50 == 0)
			cout << t << endl;
	}
}





