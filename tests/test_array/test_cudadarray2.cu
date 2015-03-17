#include <iostream>
#include <cmath>
#include "cuda_darray.h"

using namespace std;

// Test the gpu diag array
//
// Routines:
//
// sum : compute the sum of an array

#ifdef __CUDA_ARCH__
#define CUDAMEMBER __device__
#endif

#ifndef __CUDA_ARCH__
#define CUDAMEMBER
#endif

class f1 {
    public:
        CUDAMEMBER double operator()(double x, double y) const { return (1.0);};
};

int main(void)
{
    constexpr unsigned int Nx{16};
    constexpr unsigned int My{16};

    //cout << "Enter Nx: " << endl;
    //cin >> Nx;
    //cout << "Enter My: " << endl;
    //cin >> My;

    const double Lx = 1.0;
    const double Ly = 1.0;

    const double dx = Lx / ((double) Nx);
    const double dy = Ly / ((double) My);

    cuda_darray<double> darr(My, Nx);
    darr  = 0.0;

    //const cuda::slab_layout_t sl(-0.5 * Lx, dx, -0.5 * Ly, dy, My, Nx);
    //darr.op_scalar_fun<f1>(sl, 0);

    cout << "darr = \n" << darr << endl;

    cout << "max = " << darr.get_max() << endl;
    cout << "min = " << darr.get_min() << endl;
    cout << "sum = " << darr.get_sum() << endl;
    cout << "mean = "<< darr.get_mean() << endl;

    darr = 1.4;
    cout << "darr = \n" << darr << endl;

    cout << "max = " << darr.get_max() << endl;
    cout << "min = " << darr.get_min() << endl;
    cout << "sum = " << darr.get_sum() << endl;
    cout << "mean = "<< darr.get_mean() << endl;

}

