/*
 * test cuda_array arithmetic
 *
 */

#include <cuda_array4.h>
#include <iostream>

using namespace std;


// test arithmetic routines, 
// init values for arrays from python routines
// output values passed to python as well

extern "C" void real_add(double in_r1, double in_r2, double* out_sum, int My, int Nx)
{
    cout << "in_r1 = " << in_r1 << "\t" << "in_r2 = " << in_r2 << "\t" << "My = " << My << "\t" << "Nx = " << Nx << endl;
    cuda_array<double> r_arr1(1, My, Nx);
    cuda_array<double> r_arr2(1, My, Nx);

    r_arr1 = in_r1;
    r_arr2 = in_r2;

    cuda_array<double> result(r_arr1 + r_arr2);
    result.copy_device_to_host(out_sum);
}


extern "C" void real_sub(double in_r1, double in_r2, double* out_sum, int My, int Nx)
{
    cout << "in_r1 = " << in_r1 << "\t" << "in_r2 = " << in_r2 << "\t" << "My = " << My << "\t" << "Nx = " << Nx << endl;
    cuda_array<double> r_arr1(1, My, Nx);
    cuda_array<double> r_arr2(1, My, Nx);

    r_arr1 = in_r1;
    r_arr2 = in_r2;

    cuda_array<double> result(r_arr1 - r_arr2);
    result.copy_device_to_host(out_sum);
}


extern "C" void real_mul(double in_r1, double in_r2, double* out_sum, int My, int Nx)
{
    cout << "in_r1 = " << in_r1 << "\t" << "in_r2 = " << in_r2 << "\t" << "My = " << My << "\t" << "Nx = " << Nx << endl;
    cuda_array<double> r_arr1(1, My, Nx);
    cuda_array<double> r_arr2(1, My, Nx);

    r_arr1 = in_r1;
    r_arr2 = in_r2;

    cuda_array<double> result(r_arr1 * r_arr2);
    result.copy_device_to_host(out_sum);
}


extern "C" void real_div(double in_r1, double in_r2, double* out_sum, int My, int Nx)
{
    cout << "in_r1 = " << in_r1 << "\t" << "in_r2 = " << in_r2 << "\t" << "My = " << My << "\t" << "Nx = " << Nx << endl;
    cuda_array<double> r_arr1(1, My, Nx);
    cuda_array<double> r_arr2(1, My, Nx);

    r_arr1 = in_r1;
    r_arr2 = in_r2;

    cuda_array<double> result(r_arr1 / r_arr2);
    result.copy_device_to_host(out_sum);
}


