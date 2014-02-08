#include <iostream>
#include <vector>
#include "include/cuda_array3.h"
#include "include/initialize.h"
#include "include/cuda_types.h"

using namespace std;

int main(void)
{
    const int Nx = 16;
    const int My = 16;
    const int tlevs = 1;

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
    cuda_array<cuda::real_t, cuda::real_t> arr_r(1, Nx, My);
    cuda_array<CuCmplx<cuda::real_t>, cuda::real_t> arr_c(2, Nx, My / 2 + 1);

    // Plan forward and backward DFT
    err = cufftPlan2d(&plan_fw, Nx, My, CUFFT_D2Z);
    cout << "Error: " << err << "\n"; 
    // Plan forward and backward DFT
    err = cufftPlan2d(&plan_bw, Nx, My, CUFFT_Z2D);
    cout << "Error: " << err << "\n"; 

    // Estimate worksize
    cufftEstimate2d(Nx, My, CUFFT_D2Z, &dft_size);
    cout << "Estimated size: " << dft_size << " bytes\n";

    // Initialize 
    arr_r = 0.0;
    arr_c = 0.0;//make_cuDoubleComplex(0.0, 0.0);
    vector<double> initc;
    initc.push_back(0.0); initc.push_back(1.0);
    init_simple_sine(&arr_r, initc, delta_x, delta_y, xleft, ylo); 
    
    cout << "arr =\n" << arr_r << "\n";


    // Execute DFT
    cout << "Executing DFT\n";
    err = cufftExecD2Z(plan_fw, arr_r.get_array_d(), (cufftDoubleComplex*) arr_c.get_array_d(1));
    cout << "Error: " << err << "\t";
    cout << "arr_c =" << arr_c << "\n";


    // Inverse DFT
    cout << "Executing iDFT\n";
    err = cufftExecZ2D(plan_bw, (cufftDoubleComplex*) arr_c.get_array_d(1), arr_r.get_array_d());
    arr_r.normalize();
    cout << "Error: " << err << "\t";
    cout << "arr_r = " << arr_r << "\n";
    cout << "arr_c = " << arr_c << "\n";

    return(0);
}

