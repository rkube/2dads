/*
 * test_array.cu
 *
 * Test instantiation of real and complex cuda_arrays
 * Tests: 
 * * Instantiation of real and complex data types
 * * Arithmetic
 * * Output operators
 */

#include <iostream>
#include "include/cuda_array3.h"

using namespace std;

int main(void)
{
    const int Nx = 8;
    const int My = 8;
    const int tlevs = 1;
	cuda_array<double, double> r_arr1(tlevs, Nx, My);
    cuda_array<double, double> r_arr2(tlevs, Nx, My);

    cuda_array<CuCmplx<double>, double> c_arr1(tlevs, Nx, My);
    cuda_array<CuCmplx<double>, double> c_arr2(tlevs, Nx, My);

	cout << "arr1: Nx = " << r_arr1.get_nx() << ", My = " << r_arr1.get_my() << ", tlevs = " << r_arr1.get_tlevs() << "\n";

    r_arr1 = 1.0;
    r_arr2 = 2.0;

    cout << "================== Testing arithmetic =================\n";
    cout << "r_arr1 = " << r_arr1 << "\nr_arr2 = " << r_arr2 << "\n";
    cout << "r_arr1 + r_arr2 = " << (r_arr1 + r_arr2) << "\n";
    cout << "r_arr1 - r_arr2 = " << (r_arr1 - r_arr2) << "\n";
    cout << "r_arr1 * r_arr2 = " << (r_arr1 * r_arr2) << "\n";
    cout << "r_arr1 / r_arr2 = " << (r_arr1 / r_arr2) << "\n";


    cout << "================= Testing complex arithmetic =============\n";
    c_arr1 = CuCmplx<double>(3.0, 2.0);
    c_arr2 = CuCmplx<double>(0.0, 1.0);
    cout << "c_arr1 = " << c_arr1 << "\nc_arr2 = " << c_arr2 << "\n";
    cout << "c_arr1 + c_arr2 = " << (c_arr1 + c_arr2) << "\n";
    cout << "c_arr1 - c_arr2 = " << (c_arr1 - c_arr2) << "\n";
    cout << "c_arr1 * c_arr2 = " << (c_arr1 * c_arr2) << "\n";
    cout << "c_arr1 / c_arr2 = " << (c_arr1 / c_arr2) << "\n";
}


