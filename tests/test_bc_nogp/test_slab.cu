/*
 * Test Fourier Transformation
 */

#include <iostream>
#include "slab_bc.h"

using namespace std;

int main(void)
{
    constexpr size_t Nx{16};
    constexpr size_t My{16};

    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / double(Nx), 0.0, 1.0 / double(My), Nx, 0, My, 2);

    {
        slab_bc my_slab(my_geom, my_bvals);

        my_slab.initialize_dfttest(test_ns::field_t::arr1);
        my_slab.print_field(test_ns::field_t::arr1);

        my_slab.dft_r2c(test_ns::field_t::arr1, 0);
        cout << "===========================================================================================" << endl;

        my_slab.dft_c2r(test_ns::field_t::arr1, 0);
        my_slab.print_field(test_ns::field_t::arr1);
    } // Let managed memory go out of scope before calling cudaDeviceReset()
    cudaDeviceReset();
}
