/*
 * test_copy.cpp
 *
 * Created on: Dec 27, 2013
 *     Author: Ralph Kube
 *
 *  test copy and move operators for cuda_array
 *
 * cuda_array::copy(t_dst, t_src)
 * cuda_array::copy(t_dst, cuda_array src, t_src)
 * cuda_array::move(t_dst, t_src)
 *
 */

#include <iostream>
#include "include/cuda_types.h"
#include "include/cucmplx.h"
#include "include/cuda_array3.h"

using namespace std;

int main(void)
{
    const int Nx = 8;
    const int My = 8;
    const int tlevs = 3;

    cuda_array<cuda::cmplx_t, cuda::real_t> arr_1(tlevs, Nx, My);
    cuda_array<cuda::cmplx_t, cuda::real_t> arr_2(tlevs, Nx, My);

    arr_1 = CuCmplx<cuda::real_t>(4.0, 2.7);
    arr_2 = CuCmplx<cuda::real_t>(-2.0, -3.0);

    cout << "arr_1 = " << arr_1;
    cout << "arr_2 = " << arr_2;

    // Copy data in arr_2 from tlev=0 to tlev=2 
    arr_2.copy(2, 0);
    cout << "arr_2 = " << arr_2;
    // Copy data in arr_2, t=2 to arr_1, t=0
    arr_1.copy(0, arr_2, 2);
    cout << "arr_1 = " << arr_1;
    // Move data in arr_1 from t=0 to t=2
    arr_1.move(2, 0);
    cout << "arr_1 = " << arr_1;

    // Advance arr_1, zero it out and start again
    //   --advance-->
    // 0              1 
    // 1              2 
    // 2              0
    arr_1.advance();
    arr_1.copy(0, 1);
    arr_1.copy(2, 1);
    cout << "Cleared arr_1: arr_1 = " << arr_1;
    arr_1.copy(0, arr_2, 2);
    cout << "arr_1 " << arr_1;
    arr_1.move(2, 0);
    cout << "arr_1 = " << arr_1;

    // Looks ok, 2013-12-27
    return(0);
}
// End of file test_copy.cpp
