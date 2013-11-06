/*
 * test_arrayc.cpp
 *
 * Created on: Nov 6, 2013
 *      Author: rku000
 *
 */

#include <iostream>
#include "include/cuda_array2.h"

using namespace std;
typedef cufftDoubleComplex cmplx_t;
typedef double real_t;


int main(void)
{
    const int Nx = 16;
    const int My = 16;
    const int tlevs = 1;
    cuda_array<cmplx_t> arr_c(tlevs, Nx, My / 2 - 1);
    arr_c.set_all(make_cuDoubleComplex(59.4, 12.7));
    arr_c.copy_device_to_host();
    cout << "arr_c =\n" << arr_c << "\n";

    arr_c.enumerate_array_t(0);
    arr_c.copy_device_to_host();
    cout << "arr_c =\n" << arr_c << "\n";
    

}
