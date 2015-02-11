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
#include "cuda_types.h"
#include "cucmplx.h"
#include "cuda_array4.h"

using namespace std;

int main(void)
{
    unsigned int Nx = 0;
    unsigned int My = 0;
    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;
    cout << "\n";
    const int tlevs = 3;

    cuda_array<cuda::cmplx_t> arr_1(tlevs, My, Nx);
    arr_1 = CuCmplx<cuda::real_t>(4.0, 2.7);
    //cuda_array<cuda::cmplx_t, cuda::real_t> arr_2(tlevs, My, Nx);

    // test copy constructor
    cuda_array<cuda::cmplx_t> arr_2(tlevs, My, Nx);
    arr_2 = CuCmplx<cuda::real_t>(-2.0, -3.0);

    cout << "arr_1 = " << arr_1;
    cout << "arr_2 = " << arr_2;


    cout << " Copy data in arr_2 from tlev=0 to tlev=2 ==================================" << endl;
    arr_2.copy(2, 0);
    cout << "arr_2 =" << arr_2;
    cout << " Copy data in arr_2, t=2 to arr_1, t=0 =====================================" << endl;
    arr_1.copy(0, arr_2, 2);
    cout << "arr_1 =" << arr_1;
    cout << " Move data in arr_1 from t=0 -> t=1 ========================================" << endl;
    arr_1.move(1, 0);
    cout << "arr_1 = " << arr_1;

    // Advance arr_1, zero it out and start again
    //   --advance-->
    // 0              1
    // 1              2
    // 2              0
    cout << "advance arr1 ===============================================================" << endl;
    arr_1.advance();
    cout << arr_1;
    cout << "Copy data in arr_2 t=2 to arr_1 t=0 ========================================" << endl;
    arr_1.copy(0, arr_2, 2);
    cout << "arr_1 " << arr_1;
    cout << "Move data from arr_1, t=0 -> t=1 ===========================================" << endl;
    arr_1.move(1, 0);
    cout << "arr_1 = " << arr_1;
    cout << "arr_2 = " << arr_2;

    // Looks ok, 2013-12-27
    // Looks ok, 2015-02-09


    cout << "arr_1 += arr_2  ============================================================" << endl;
    arr_1 -= arr_2;
    cout << "arr_1 = " << arr_1;
    return(0);
}
// End of file test_copy.cpp
