/*
 * test array enumeration
 */

#include <iostream>
#include <cuda_array4.h>

using namespace std;

int main(void)
{
    const unsigned int t = 1;
    unsigned int My;
    unsigned int Nx;
    cout << "Enter Nx:" << endl;
    cin >> Nx;
    cout << "Enter My:" << endl;
    cin >> My;

    cout << "float:=====" << endl;
    cuda_array<float> f_arr(t, My, Nx);
    f_arr.enumerate_array(0);
    cout << f_arr << endl;

    cout << "double:=====" << endl;
    cuda_array<double> d_arr(t, My, Nx);
    d_arr.enumerate_array(0);
    cout << d_arr << endl;

    cout << "int:=====" << endl;
    cuda_array<int> i_arr(t, My, Nx);
    i_arr.enumerate_array(0);
    cout << i_arr << endl;

    cout << "cucmplx:=====" << endl;
    cuda_array<CuCmplx<double> > c_arr(t, My, Nx);
    c_arr.enumerate_array(0);
    cout << c_arr << endl;
}
