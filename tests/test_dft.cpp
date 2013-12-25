#include <iostream>
#include <vector>
#include "include/cuda_array2.h"
#include "include/initialize.h"
#include "include/cuda_types.h"

using namespace std;

int main(void)
{
    const int Nx = 16;
    const int My = 16;
    const int tlevs = 2;

    // Slab layout
    const double xleft = -1.0;
    const double ylo = -1.0;
    const double Lx = 2.0;
    const double Ly = 2.0;
    const double delta_x = Lx / double(Nx);
    const double delta_y = Ly / double(My);


    cufftHandle plan_fw;
    cufftHandle plan_bw;
    cufftResult err;
    size_t dft_size;

    // Create input arrays
    cuda_array<cuda::real_t> arr_r(tlevs, Nx, My);
    cuda_array<cuda::cmplx_t> arr_c(tlevs, Nx, My / 2 + 1);

    // Plan forward and backward DFT
    err = cufftPlan2d(&plan_fw, Nx, My, CUFFT_D2Z);
    cout << "Error: " << err << "\n"; 

    // Estimate worksize
    cufftEstimate2d(Nx, My, CUFFT_D2Z, &dft_size);
    cout << "Estimated size: " << dft_size << " bytes\n";

    // Initialize 
    arr_r = 0.0;
    arr_c = make_cuDoubleComplex(0.0, 0.0);
    vector<double> initc = {2.0, 0.0};
    init_simple_sine(&arr_r, initc, delta_x, delta_y, xleft, ylo); 
    arr_r.copy_device_to_host();
    cout << "arr =\n" << arr_r << "\n";


    // Execute DFT
    cout << "Executing DFT\n";
    err = cufftExecD2Z(plan_fw, arr_r.get_array_d(0), arr_c.get_array_d(1));
    cout << "Error: " << err << "\n";

    arr_c.copy_device_to_host();
    cout << "arr_c =\n" << arr_c << "\n";

    return(0);
}

