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
#include "derivatives.h"

typedef CuCmplx<double> cmplx_t;

#define ELEM_PER_THREAD_T 1


using namespace std;



int main(void)
{
	constexpr int Nx{64};
	constexpr int My{64};
	constexpr double Lx{10.0};
	constexpr double Ly{10.0};
	constexpr double dx{Lx / Nx};
	constexpr double dy{Ly / My};

    cout << "Grid: Lx = " << Lx << ", Nx = " << Nx << ", dx = " << dx << endl;

    cuda::slab_layout_t sl{-0.5 * Lx, dx, -0.5 * Ly, dy, My, Nx};

    // Create derivs object
    derivs<double> der(sl);


	cuda_array<double> r_arr(1, My, Nx);
    cuda_array<double> r_arr_x(1, My, Nx);
	cuda_array<double> r_arr_y(1, My, Nx);

    cuda_array<double> r_arr_x2(1, My, Nx);
	cuda_array<double> r_arr_y2(1, My, Nx);

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

    der.d_dx1_dy1(r_arr, r_arr_x, r_arr_y);

    der.d_dx2_dy2(r_arr, r_arr_x2, r_arr_y2);

//
//	// Initialize cufft
//	cufftResult err;
//    cufftHandle plan_r2c;
//    cufftHandle plan_c2r;
//    err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z);
//    err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D);
//
//	err = cufftExecD2Z(plan_r2c, r_arr.get_array_d(), (cufftDoubleComplex*) c_arr.get_array_d(0));
//	if(err != CUFFT_SUCCESS)
//		throw;
//
//	// Method 1... generate k-map and run derivs in one kernel call
//	gen_k_map<My, Nx21><<<dim3(1, My), dim3(Nx21, 1)>>>(kmap.get_array_d(), two_pi_Lx, two_pi_Ly);
//	gpuErrchk(cudaPeekAtLastError());
//
////	void d_dx_dy_map(cmplx_t*  in, cmplx_t*  out_x, cmplx_t*  out_y, cmplx_t*  kmap)
//
//	//d_dx_dy_map<My, Nx21, elem_per_thread><<<gridsize, blocksize>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), kmap.get_array_d());
//	d_dx_dy_map_sh<My, Nx21, elem_per_thread><<<gridsize, blocksize, shmem_size>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), kmap.get_array_d());
//
//	gpuErrchk(cudaPeekAtLastError());
//
//
//	// Transform to real space
//	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr.get_array_d(), r_arr.get_array_d());
//	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr_x.get_array_d(), r_arr_x.get_array_d());
//	err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) c_arr_y.get_array_d(), r_arr_y.get_array_d());
//
//	r_arr.normalize();
//	r_arr_x.normalize();
//	r_arr_y.normalize();
//
//
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

	of.open("r_arr_x2.dat");
	of << r_arr_x2;
	of.close();

	of.open("r_arr_y2.dat");
	of << r_arr_y2;
	of.close();

/*
	for(int t = 0; t < 1000; t++)
	{
	    d_dx_dy_map_sh<My, Nx21, elem_per_thread><<<gridsize, blocksize, shmem_size>>>(c_arr.get_array_d(), c_arr_x.get_array_d(), c_arr_y.get_array_d(), kmap.get_array_d());
		if (t % 50 == 0)
			cout << t << endl;
	}
*/
}



