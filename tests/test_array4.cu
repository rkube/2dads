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
#include "cuda_array4.h"

using namespace std;

int main(void)
{
    int Nx = 0;
    int My = 0;
    const int tlevs = 1;
    cout << "Enter Nx:";
    cin >> Nx;
    cout << "Enter My:";
    cin >> My;
    cout << "\n"; 

    cuda_array<double> r_arr1(tlevs, My, Nx);
    cuda_array<double> r_arr2(tlevs, My, Nx);
    cuda_array<CuCmplx<double> > c_arr1(tlevs, My, Nx);
    cuda_array<CuCmplx<double> > c_arr2(tlevs, My, Nx);

    cout << "arr1: Nx = " << r_arr1.get_nx() << ", My = " << r_arr1.get_my() << ", tlevs = " << r_arr1.get_tlevs() << "\n";
    cout << "arr2: Nx = " << r_arr2.get_nx() << ", My = " << r_arr2.get_my() << ", tlevs = " << r_arr2.get_tlevs() << "\n";

    cout << "==============================================\n";
    cout << "blockDim = (" << r_arr1.get_block().x << ", " << r_arr1.get_block().y << ")\n";
    cout << "gridDim = (" << r_arr1.get_grid().x << ", " << r_arr1.get_grid().y << ")\n";
    cout << "================== Enumerating array ==================\n";
    r_arr1.enumerate_array(0);
    cout << r_arr1 << "\n\n";

    // Test real value arithmetic
    r_arr1 = 1.0;
    r_arr2 = 3.0;
    double val = 3.0;

    cout << "================== Testing real scalar arithmetic =================\n";
    cout << "r_arr1 = " << r_arr1 << "\nr_arr2 = " << r_arr2 << "\n";
    cout << "r_arr1 + " << val << " = " << (r_arr1 + val) << "\n";
    cout << "r_arr1 - " << val << " = " << (r_arr1 - val) << "\n";
    cout << "r_arr1 * " << val << " = " << (r_arr1 * val) << "\n";
    cout << "r_arr1 / " << val << " = " << (r_arr1 / val) << "\n";

    cout << "================== Testing real array arithmetic =================\n";
    cout << "r_arr1 + r_arr2 = " << (r_arr1 + r_arr2) << "\n";
    cout << "r_arr1 - r_arr2 = " << (r_arr1 - r_arr2) << "\n";
    cout << "r_arr1 * r_arr2 = " << (r_arr1 * r_arr2) << "\n";
    cout << "r_arr1 / r_arr2 = " << (r_arr1 / r_arr2) << "\n";


    cout << "================= Testing complex array arithmetic =============\n";
    c_arr1 = CuCmplx<double>(3.0, 2.0);
    c_arr2 = CuCmplx<double>(0.0, 1.0);
    CuCmplx<double> cval(0.0, 1.0);
    cout << "================= Testing complex scalar arithmetic ============\n";
    cout << "c_arr1 = " << c_arr1 << "\ncval = " << cval << "\n";
    cout << "c_arr1 + cval = " << (c_arr1 + cval) << "\n";
    cout << "c_arr1 - cval = " << (c_arr1 - cval) << "\n";
    cout << "c_arr1 * cval = " << (c_arr1 * cval) << "\n";
    cout << "c_arr1 / cval = " << (c_arr1 / cval) << "\n";

    cout << "================= Testing complex scalar arithmetic ============\n";
    cout << "c_arr1 = " << c_arr1 << "\nc_arr2 = " << c_arr2 << "\n";
    cout << "c_arr1 + c_arr2 = " << (c_arr1 + c_arr2) << "\n";
    cout << "c_arr1 - c_arr2 = " << (c_arr1 - c_arr2) << "\n";
    cout << "c_arr1 * c_arr2 = " << (c_arr1 * c_arr2) << "\n";
    cout << "c_arr1 / c_arr2 = " << (c_arr1 / c_arr2) << "\n";
}


