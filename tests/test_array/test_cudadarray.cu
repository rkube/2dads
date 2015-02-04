#include <iostream>
#include <cmath>
#include "cuda_darray.h"

using namespace std;

// Test the gpu diag array
//
// Routines:
//
// bar:   get poloidal mean
// tilde: get poloidal fluctuation around mean

//template __device__ __host__ double d_op_add<double>(double, double);
//template __device__ __host__ double d_op_max<double>(double, double);
//template __device__ __host__ double d_op_min<double>(double, double);
//
//typedef double (*op_func_t) (double, double);
//
//__constant__ op_func_t op_func_table[] = {d_op_add, d_op_max, d_op_min};

int main(void)
{
    const unsigned int Nx = 16;
    const unsigned int My = 16;

    const double Lx = 1.0;
    const double Ly = 1.0;

    const double dx = 2. * Lx / ((double) Nx);
    const double dy = 2. * Ly / ((double) My);

    cuda_array<double, double> carr(1, My, Nx);
    cuda_darray<double> darr(My, Nx);

    double x = 0.0;
    double y = 0.0;
    for(uint m = 0; m < carr.get_my(); m++)
    {
        y = -Ly + ((double) m) * dy;
        for(uint n = 0; n < carr.get_nx(); n++)
        {
            x = -Lx + ((double) n) * dx;
            carr(0, m, n) = 0.3 * x + sin(2.0 * M_PI * y);
        }
    }
    cout << "Copy host to device" << endl;
    carr.copy_host_to_device();
    cout << "darr = carr" << endl;
    darr = carr;
    cout << darr << "\n";

    darr.reduce_profile();
    darr.reduce_max();
    darr.reduce_mean();
    darr.reduce_min();

    cout << "max = " << darr.get_max() << endl;
    cout << "min = " << darr.get_min() << endl;
    cout << "mean = " << darr.get_mean() << endl;
    //darr.get_mean()
}
