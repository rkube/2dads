#include <iostream>
#include <cmath>
#include "cuda_darray.h"

using namespace std;

// Test L2 routine of gpu_diag array

class fun1 {
    // f(x, y) = exp(-x^2 - y^2)
    public:
        CUDAMEMBER  double operator() (double x,  double y) const { return ( exp(-x * x  - y * y) );};
};

int main(void)
{
    constexpr int Nx{16};
    constexpr int My{16};

    constexpr double Lx{2.0};
    constexpr double Ly{2.0};

    constexpr double dx{Lx / Nx};
    constexpr double dy{Ly / My};

    const cuda::slab_layout_t sl(-0.5 * Lx, Lx / (cuda::real_t) Nx, -0.5 * Ly, Ly / (cuda::real_t) My, My, Nx);


    cuda_darray<cuda::real_t> arr(My, Nx);
    arr.op_scalar_fun<fun1> (sl, 0);

    cout << arr << endl;

    double sum{0.0};
    for(int m = 0; m < My; m++)
        for(int n = 0; n < Nx; n++)
            sum += arr(m, n) * arr(m, n);

    cout << "L2 norm is " << arr.get_L2() << endl;
    cout << "L2 norm (host) = " << sqrt(sum) / double(Nx * My);


    return(0);
}
