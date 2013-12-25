#include <iostream>
#include <vector>
#include "include/initialize.h"
//#include "include/cuda_types.h"
//#include "include/cuda_array2.h"

using namespace std;

int main(void)
{

    const int Nx = 16;
    const int My = 16;
    const int tlevs = 1;

    const double xleft = -1.0;
    const double ylo = -1.0;
    const double Lx = 2.0;
    const double Ly = 2.0;
    const double delta_x = Lx / double(Nx);
    const double delta_y = Ly / double(My);

    vector<double> initc = {0.0, 2.0};
    cuda_array<cuda::real_t> arr(tlevs, Nx, My);
    arr = 0.0;
    cout << "xleft=" << xleft << ", delta_x=" << delta_x << ", Nx=" << Nx << "\n";
    cout << "ylo=" << ylo << ", delta_y=" << delta_y << ", Ny=" << My << "\n";
    cout << "arr =\n" << arr << "\n";

    init_simple_sine(&arr, initc, delta_x, delta_y, xleft, ylo);
    //arr.set_all(2.2);
    arr.copy_device_to_host();
    cout << "arr =\n" << arr << "\n";

    return(0);
}


