/*
 * Test upcast member functions of cuda_darray
 */


#include <iostream>
#include "cuda_darray.h"

using namespace std;

int main(void)
{
    constexpr int Nx{21};
    constexpr int My{9};

    constexpr double Lx{1.0};
    constexpr double Ly{1.0};

    constexpr double dx{2.0 * Lx / double(Nx)};
    constexpr double dy{2.0 * Ly / double(My)};

    double* x_arr = new double[Nx];
    double* y_arr = new double[My];

    for(int n = 0; n < Nx; n++)
        x_arr[n] = -Lx + (double) n * dx;
    for(int m = 0; m < My; m++)
        y_arr[m] = - Ly + (double) m * dy;

//    for(int n = 0; n < Nx; n++)
//        cout << x_arr[n]<< " ";


    cuda_darray<double> arr(My, Nx);

    arr.upcast_col(x_arr, Nx);
    cout << "arr = " << arr << endl;

    arr.upcast_row(y_arr, My);
    cout << "arr = " << arr << endl;


}

