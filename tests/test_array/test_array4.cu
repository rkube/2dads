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
    int Nx{0};
    int My{0};
    constexpr int tlevs{4};
    cout << "Enter Nx:";
    cin >> Nx;
    cout << "Enter My:";
    cin >> My;
    cout << "\n"; 

    cuda_array<double> r_arr1(tlevs, My, Nx);
    
//    cuda_array<double> r_arr2(tlevs, My, Nx);
//    cuda_array<CuCmplx<double> > c_arr1(tlevs, My, Nx);
//    cuda_array<CuCmplx<double> > c_arr2(tlevs, My, Nx);

//    const double rval1{1.0};
//    const double rval2{3.0};

//    const CuCmplx<double> cval1(3.0, 2.0);
//    const CuCmplx<double> cval2(0.0, 1.0);

//    cout << "arr1: Nx = " << r_arr1.get_nx() << ", My = " << r_arr1.get_my() << ", tlevs = " << r_arr1.get_tlevs() << "\n";
//    cout << "arr2: Nx = " << r_arr2.get_nx() << ", My = " << r_arr2.get_my() << ", tlevs = " << r_arr2.get_tlevs() << "\n";

//    cout << "==============================================" <<endl;
//    cout << "blockDim = (" << r_arr1.get_block().x << ", " << r_arr1.get_block().y << ")" << endl;
//    cout << "gridDim = (" << r_arr1.get_grid().x << ", " << r_arr1.get_grid().y << ")" << endl;
//    cout << "================== Enumerating array ==================" << endl;
//    r_arr1.enumerate_array(0);
    //cout << r_arr1 << "\n\n";

    // Test real value arithmetic
//    r_arr1 = rval1;
//    r_arr2 = rval2;

// calling op_scalar_t performs scalar operation on the array data in place
// calling op_scalar_t creates no new cuda_array, but modifies the data of the array
//    cout << "================== Testing real scalar arithmetic using op_scalar_t() =================" << endl;
//    cout << "r_arr1 = " << r_arr1 << endl;
//
//    r_arr1.op_scalar_t<d_op1_addassign<double> >(rval1, 0);
//    cout << "r_arr1.op_scalar_t<d_op1_addassign>(" << rval1 << ")" << endl;
//    cout << "r_arr1 = " << r_arr1 << endl;
//
//    r_arr1.op_scalar_t<d_op1_assign<double> >(rval1, 0);
//    r_arr1.op_scalar_t<d_op1_subassign<double> >(rval2, 0);
//    cout << "r_arr1.op_scalar_t<d_op1_subassign>(" << rval2 << ")" << endl;
//    cout << "r_arr1 = " << r_arr1 << endl;
//
//    r_arr1.op_scalar_t<d_op1_assign<double> >(rval1, 0);
//    r_arr1.op_scalar_t<d_op1_mulassign<double> >(rval2, 0);
//    cout << "r_arr1.op_scalar_t<d_op1_mulassign>(" << rval2 << ")" << endl;
//    cout << "r_arr1 = " << r_arr1 << endl;
//
//    r_arr1.op_scalar_t<d_op1_assign<double> >(rval1, 0);
//    r_arr1.op_scalar_t<d_op1_divassign<double> >(rval2, 0);
//    cout << "r_arr1.op_scalar_t<d_op1_divassign>(" << rval2 << ")" << endl;
//    cout << "r_arr1 = " << r_arr1 << endl;


// operator[+-*/] perform out-of-place operation on array data
// returns a new cuda_array
//    cout << "================== Testing real array arithmetic using operator[+-*/] =================" << endl;

//    r_arr1 = rval1;
//    cuda_array<double> result(r_arr1 + rval1);
//    cout << "r_arr1 + " << rval1 << " = " << result << "\n";
//    result = r_arr1 - rval1;
//    cout << "r_arr1 - " << rval1 << " = " << result << "\n";
//    result = r_arr1 * rval1;
//    cout << "r_arr1 * " << rval1 << " = " << result << "\n";
//    result = r_arr1 / rval1;
//    cout << "r_arr1 / " << rval1 << " = " << result << "\n";

//    cout << "================== Testing real array arithmetic =================\n";
//    r_arr2 = rval2;
//    result = r_arr1 + r_arr2;
//    cout << "r_arr1 + r_arr2 = " << result << "\n";
//    result = r_arr1 - r_arr2;
//    cout << "r_arr1 - r_arr2 = " << result << "\n";
//    result = r_arr1 * r_arr2;
//    cout << "r_arr1 * r_arr2 = " << result << "\n";
//    result = r_arr1 / r_arr2;
//    cout << "r_arr1 / r_arr2 = " << result << "\n";
//
//
//    cout << "================= Testing complex array arithmetic =============" << endl;

//    c_arr1 = cval1;
//    c_arr2 = cval2;

//    cout << "c_arr1 = " << c_arr1 << endl;
//    cout << "c_arr2 = " << c_arr2 << endl;
    //cout << "cval = " << cval1 << endl;

//    cout << "=================== Testing in-place arithmetic with scalar RHS======" << endl;
//    c_arr1.op_scalar_t<d_op1_assign<CuCmplx<double> > >(cval1, 0);
//    cout << "c_arr1.op_scalar_t<d_op1_assign>(" << cval1 << ")" << endl;
//    cout << "c_arr1 = " << c_arr1 << endl;
//
//    c_arr1.op_scalar_t<d_op1_assign<CuCmplx<double> > >(cval1, 0);
//    c_arr1 += cval2;
//    //c_arr1.op_scalar_t<d_op1_addassign<CuCmplx<double> > >(cval2, 0);
//    cout << "c_arr1.op_scalar_t<d_op1_addassign>(" << cval2 << ")" << endl;
//    cout << "c_arr1 = " << c_arr1 << endl;
//
//    c_arr1.op_scalar_t<d_op1_assign<CuCmplx<double> > >(cval1, 0);
//    c_arr1 -= cval2;
//    //c_arr1.op_scalar_t<d_op1_subassign<CuCmplx<double> > >(cval2, 0);
//    cout << "c_arr1.op_scalar_t<d_op1_subassign>(" << cval2 << ")" << endl;
//    cout << "c_arr1 = " << c_arr1 << endl;
//
//    c_arr1.op_scalar_t<d_op1_assign<CuCmplx<double> > >(cval1, 0);
//    c_arr1 *= cval2;
//    //c_arr1.op_scalar_t<d_op1_mulassign<CuCmplx<double> > >(cval2, 0);
//    cout << "c_arr1.op_scalar_t<d_op1_mulassign>(" << cval2 << ")" << endl;
//    cout << "c_arr1 = " << c_arr1 << endl;
//
//    c_arr1.op_scalar_t<d_op1_assign<CuCmplx<double> > >(cval1, 0);
//    c_arr1 /= cval2;
//    //c_arr1.op_scalar_t<d_op1_divassign<CuCmplx<double> > >(cval2, 0);
//    cout << "c_arr1.op_scalar_t<d_op1_divassign>(" << cval2 << ")" << endl;
//    cout << "c_arr1 = " << c_arr1 << endl;


    //cuda_array<CuCmplx<double> > result (c_arr1 + c_arr2);

//    cout << "================= Testing complex array arithmetic ============" << endl;
//    cout << "c_arr1 = " << c_arr1 << "\nc_arr2 = " << c_arr2 << endl;
//    cout << "c_arr1 + c_arr2 = " << result << endl;
//
//    result = c_arr1 - c_arr2;
//    cout << "c_arr1 - c_arr2 = " << result << endl;
//
//    result = c_arr1 * c_arr2;
//    cout << "c_arr1 * c_arr2 = " << result << endl;
//
//    result = c_arr1 / c_arr2;
//    cout << "c_arr1 / c_arr2 = " << result << endl;
//    cout << "exiting" << endl;
    return(0);
}


